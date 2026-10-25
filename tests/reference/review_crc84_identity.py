"""Explicit CRC84 diagram transcriptions; no automatic identity or alias resolver."""

import argparse
import json
import re
import subprocess
import time
import traceback
from pathlib import Path

from chemworld.physchem.elements import parse_formula

ROUTE = "reviewed_CRC84_table_and_diagram"
URL = "https://fap.if.usp.br/~jhsevero/Fisica_Experimental_C_Semestral_2022/crc.pdf"
# CAS, entry, PDF table page, source name, formula, printed MW, scope, centres.
ENTRIES = [
    ("111675-77-5", 7289, 503, "2-Methyl-1-heptanol, (±)", "C8H18O", 130.228, "RACEMIC", 1),
    ("111767-95-4", 7292, 503, "5-Methyl-1-heptanol, (±)", "C8H18O", 130.228, "RACEMIC", 1),
    ("100296-26-2", 7299, 505, "2-Methyl-3-heptanol, (±)", "C8H18O", 130.228, "RACEMIC", 1),
    ("100295-85-0", 7303, 505, "6-Methyl-3-heptanol, (±)", "C8H18O", 130.228, "RACEMIC", 1),
    ("111768-04-8", 7331, 505, "2-Methyl-1-hexanol, (±)", "C7H16O", 116.201, "RACEMIC", 1),
    ("55156-16-6", 7571, 519, "3-Methyl-2-pentanone, (±)", "C6H12O", 100.158, "RACEMIC", 1),
    ("772-17-8", 7003, 489, "β-Methylbenzenepropanoic acid, (±)",
     "C10H12O2", 164.201, "RACEMIC", 1),
    ("17556-48-8", 10420, 667, "1,1,1-Trifluoro-2-propanol, (±)",
     "C3H5F3O", 114.066, "RACEMIC", 1),
    ("5183-77-7", 2927, 275, "trans-1,2-Dibromocyclohexane, (±)",
     "C6H10Br2", 241.951, "RACEMIC", 2),
    ("57968-71-5", 3533, 307, "Diethyl DL-tartrate", "C8H14O6", 206.193, "RACEMIC", 2),
    ("54638-10-7", 8663, 575, "(R)-sec-Pentyl acetate", "C7H14O2", 130.185, "ABSOLUTE", 1),
    ("10395-45-6", 1786, 217, "Carvenone, (S)", "C10H16O", 152.233, "ABSOLUTE", 1),
    ("13454-99-4", 1661, 770, "Phosphorus(V) dichloride trifluoride",
     "PCl2F3", 158.874, "NOT_SPECIFIED_BY_SOURCE", 0),
]
INITIAL_CAS = tuple(r[0] for r in ENTRIES)
ADDITIONAL_ENTRIES = [
    ("2512-81-4", 4266, 347, "1,2-Dimethylpiperidine, (±)", "C7H15N", 113.201, "RACEMIC", 1),
    ("6195-92-2", 7171, 497, "3-Methylcyclopentanone, (±)", "C6H10O", 98.142, "RACEMIC", 1),
    ("42569-16-4", 8649, 573, "3-Penten-2-ol, (±)", "C5H10O", 86.132, "RACEMIC", 1),
    ("20089-07-0", 7069, 493, "2-Methyl-1-butanethiol, (+)", "C5H12S", 104.214, "ABSOLUTE", 1),
]
ADDITIONAL_CAS = tuple(r[0] for r in ADDITIONAL_ENTRIES)
ENTRIES += ADDITIONAL_ENTRIES
BY_CAS = {r[0]: r for r in ENTRIES}


def read_source(directory, selected=None):
    assert (directory / "crc84.pdf").read_bytes().startswith(b"%PDF-")
    pages = (directory / "crc84.txt").read_text(encoding="utf8").split("\f")
    sources = {}
    for cas, entry, page, name, formula, weight, stereo, centres in ENTRIES:
        if selected is not None and cas not in selected:
            continue
        match = re.search(rf"^\s*{entry}\s+(.+?)(?=^\s*{entry + 1}\s+|\Z)",
                          pages[page - 1], re.M | re.S)
        assert match is not None
        text = match[1]
        assert re.findall(r"\b\d{2,7}-\d{2}-\d\b", text) == [cas]
        assert re.search(rf"{cas}\s+{weight:.3f}\b", text)
        if stereo == "RACEMIC":
            assert "(±)" in text or (cas == "57968-71-5" and "DiethylDL-tartrate" in text)
        if cas == "54638-10-7":
            assert "sec-Pentyl acetateR)(" in text
            assert "sec-Pentylacetate" in pages[page].replace(" ", "")
        elif cas == "772-17-8":
            assert "b-Methylbenzenepropanoic acid" in text
        else:
            expected = name.replace(" ", "")
            assert expected in text.replace(" ", "")
        if centres:
            # Name extraction checks the facing diagram page, in addition to visual review.
            stem = name.split(", (")[0].replace("β", "b")
            if cas == "57968-71-5":
                assert "DL-tartrate" in pages[page] and "Diethyl" in pages[page]
            elif cas not in {"54638-10-7", "772-17-8"}:
                assert stem.replace(" ", "") in pages[page].replace(" ", "")
        sources[cas] = {
            "CAS": cas, "entry": entry, "page": page,
            "diagram_page": page + 1 if centres else None,
            "name": name, "formula": formula, "molecular_weight_g_mol": weight,
            "stereochemistry": stereo, "stereo_centers": centres,
            "url": URL, "edition": "CRC Handbook 84th edition, 2003-2004",
            "identity_only": True,
        }
    return sources


def diagram_mol(cas):
    """Transcribed connectivity plus explicitly documented stereo encoding choices.

    Single-centre (+/-) graphs use one arbitrary representative, then official SRac.
    Tartrate DL denotes RR/SS (not meso); both OH wedges give RR in this layout.
    For carvenone, CIP order is carbonyl ring path > other ring path > methyl > H;
    dashed methyl in the specified hexagon gives S. For pentyl acetate the order is
    O > propyl > methyl > H; dashed methyl matches the source drawing's R label.
    """
    chain_settings = {
        "111675-77-5": (7, 2, 1), "111767-95-4": (7, 5, 1),
        "100296-26-2": (7, 2, 3), "100295-85-0": (7, 6, 3),
        "111768-04-8": (6, 2, 1), "55156-16-6": (5, 3, 2),
    }
    if cas in chain_settings:
        length, methyl, oxygen = chain_settings[cas]
        atoms = [(float(i), 0.5 * (i % 2), "C") for i in range(length)]
        bonds = [(i, i + 1, 1, 0) for i in range(1, length)]
        mx, my, _ = atoms[methyl - 1]
        ox, oy, _ = atoms[oxygen - 1]
        atoms += [(mx, my + (1.2 if my else -1.2), "C")]
        atoms += [(ox - 0.8, oy - 0.5, "O") if oxygen == 1 else
                  (ox, oy + (1.2 if oy else -1.2), "O")]
        centre_on_methyl = oxygen == 1 or cas == "55156-16-6"
        bonds += [(methyl, length + 1, 1, 1 if centre_on_methyl else 0),
                  (oxygen, length + 2, 2 if cas == "55156-16-6" else 1,
                   0 if centre_on_methyl else 1)]
    elif cas == "772-17-8":
        atoms = [(0, 0, "C"), (1, .5, "C"), (2, 0, "C"), (2, -1, "C"),
                 (3, .5, "C"), (3, 1.5, "C"), (4, 2, "C"), (5, 1.5, "C"),
                 (5, .5, "C"), (4, 0, "C"), (-.7, .7, "O"), (-.7, -.7, "O")]
        bonds = [(1, 2, 1, 0), (2, 3, 1, 0), (3, 4, 1, 1), (3, 5, 1, 0),
                 (5, 6, 2, 0), (6, 7, 1, 0), (7, 8, 2, 0), (8, 9, 1, 0),
                 (9, 10, 2, 0), (10, 5, 1, 0), (1, 11, 2, 0), (1, 12, 1, 0)]
    elif cas == "17556-48-8":
        atoms = [(0, 0, "C"), (1, .5, "C"), (2, 0, "C"), (1, 1.5, "O"),
                 (2, -1, "F"), (3, 0, "F"), (2.7, .7, "F")]
        bonds = [(1, 2, 1, 0), (2, 3, 1, 0), (2, 4, 1, 1),
                 (3, 5, 1, 0), (3, 6, 1, 0), (3, 7, 1, 0)]
    elif cas == "5183-77-7":
        atoms = [(0, 1, "C"), (.866, .5, "C"), (.866, -.5, "C"),
                 (0, -1, "C"), (-.866, -.5, "C"), (-.866, .5, "C"),
                 (0, 1.8, "Br"), (1.559, .9, "Br")]
        bonds = [(i, i % 6 + 1, 1, 0) for i in range(1, 7)]
        bonds += [(1, 7, 1, 1), (2, 8, 1, 6)]
    elif cas == "57968-71-5":
        atoms = [(0, 0, "C"), (1, .5, "C"), (-1, .5, "C"), (2, 0, "C"),
                 (0, -1, "O"), (1, 1.5, "O"), (-1, 1.5, "O"), (2, -1, "O"),
                 (-2, 0, "O"), (3, .5, "O"), (-3, .5, "C"), (-4, 0, "C"),
                 (4, 0, "C"), (5, .5, "C")]
        bonds = [(1, 2, 1, 0), (1, 3, 1, 0), (2, 4, 1, 0),
                 (1, 5, 1, 1), (2, 6, 1, 1), (3, 7, 2, 0), (4, 8, 2, 0),
                 (3, 9, 1, 0), (4, 10, 1, 0), (9, 11, 1, 0), (11, 12, 1, 0),
                 (10, 13, 1, 0), (13, 14, 1, 0)]
    elif cas == "54638-10-7":
        atoms = [(0, 0, "C"), (-.866, -.5, "O"), (.866, -.5, "C"),
                 (1.732, 0, "C"), (2.598, -.5, "C"), (0, 1, "C"),
                 (-1.732, 0, "C"), (-1.732, 1, "O"), (-2.598, -.5, "C")]
        bonds = [(1, 2, 1, 0), (1, 3, 1, 0), (3, 4, 1, 0), (4, 5, 1, 0),
                 (1, 6, 1, 6), (2, 7, 1, 0), (7, 8, 2, 0), (7, 9, 1, 0)]
    elif cas == "10395-45-6":
        atoms = [(0, 1, "C"), (.866, .5, "C"), (.866, -.5, "C"),
                 (0, -1, "C"), (-.866, -.5, "C"), (-.866, .5, "C"),
                 (0, 1.8, "C"), (1.559, .9, "O"), (0, -1.8, "C"),
                 (-.7, -2.2, "C"), (.7, -2.2, "C")]
        bonds = [(i, i % 6 + 1, 2 if i == 3 else 1, 0) for i in range(1, 7)]
        bonds += [(1, 7, 1, 6), (2, 8, 2, 0), (4, 9, 1, 0), (9, 10, 1, 0), (9, 11, 1, 0)]
    elif cas == "2512-81-4":
        atoms = [(0, 1, "C"), (.866, .5, "C"), (.866, -.5, "C"),
                 (0, -1, "N"), (-.866, -.5, "C"), (-.866, .5, "C"),
                 (0, -1.8, "C"), (1.559, -.9, "C")]
        bonds = [(i, i % 6 + 1, 1, 0) for i in range(1, 7)]
        bonds += [(4, 7, 1, 0), (3, 8, 1, 1)]
    elif cas == "6195-92-2":
        atoms = [(0, 1, "C"), (.951, .309, "C"), (.588, -.809, "C"),
                 (-.588, -.809, "C"), (-.951, .309, "C"), (0, 1.8, "O"),
                 (1.1, -1.5, "C")]
        bonds = [(i, i % 5 + 1, 1, 0) for i in range(1, 6)]
        bonds += [(1, 6, 2, 0), (3, 7, 1, 1)]
    elif cas == "42569-16-4":
        # The printed zigzag fixes E: CH(OH)CH3 and CH3 lie on opposite sides.
        atoms = [(0, 0, "C"), (1, .5, "C"), (2, 0, "C"), (3, .5, "C"),
                 (4, 0, "C"), (1, 1.5, "O")]
        bonds = [(1, 2, 1, 0), (2, 3, 1, 0), (3, 4, 2, 0), (4, 5, 1, 0), (2, 6, 1, 1)]
    elif cas == "20089-07-0":
        # Copy the printed solid methyl wedge. CIP CH2SH > ethyl > methyl > H gives S.
        atoms = [(-2, 0, "C"), (-1, .5, "C"), (0, 0, "C"), (1, .5, "C"),
                 (2, 0, "S"), (0, -1, "C")]
        bonds = [(1, 2, 1, 0), (2, 3, 1, 0), (3, 4, 1, 0), (4, 5, 1, 0), (3, 6, 1, 1)]
    else:
        raise ValueError(cas)
    lines = [f"CRC84 {BY_CAS[cas][1]} {cas}", "  ChemWorld 2D", "",
             f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0            999 V2000"]
    lines += [f"{x:10.4f}{y:10.4f}{0.0:10.4f} {element:<3}"
              " 0  0  0  0  0  0  0  0  0  0  0  0" for x, y, element in atoms]
    lines += [f"{a:3d}{b:3d}{order:3d}{stereo:3d}  0  0  0"
              for a, b, order, stereo in bonds]
    return "\n".join([*lines, "M  END", ""])


def read_documents(source_directory, encoding_directory, additional_directory=None):
    directories = [encoding_directory] + ([additional_directory] if additional_directory else [])
    all_rows = []
    for directory in directories:
        encoded = json.loads((directory / "summary.json").read_text(encoding="utf8"))
        expected = 22 if directory == encoding_directory else 7
        assert encoded["reference_version"] == "InChI 1.07.5"
        assert encoded["total"] == encoded["encoded"] == len(encoded["results"]) == expected
        assert encoded["failed"] == 0 and encoded["setup_failure"] is None
        all_rows += [(directory, r) for r in encoded["results"]]
    selected = set(INITIAL_CAS) | (set(ADDITIONAL_CAS) if additional_directory else set())
    sources = read_source(source_directory, selected)
    documents = {cas: {"source": source, "encoding": {}} for cas, source in sources.items()}
    for directory, row in all_rows:
        cas, mode = row["source_CAS"], row["mode"]
        assert (directory / f"{cas}.mol").read_text(encoding="utf8") == diagram_mol(cas)
        raw = (directory / f"{cas}-{mode}.txt").read_text(encoding="utf8")
        assert re.findall(r"^InChI=.+$", raw, re.M) == [row["inchi"]]
        assert re.findall(r"^InChIKey=(.+)$", raw, re.M) == [row["inchi_key"]]
        assert row["returncode"] == 0 and row["status"] == "encoded"
        assert row["options"] == ["/Key", "/AuxNone", "/NoLabels"] + (
            ["/SRac"] if mode == "racemic" else []
        )
        assert mode not in documents[cas]["encoding"]
        documents[cas]["encoding"][mode] = row
    for cas, document in documents.items():
        source, encodings = document["source"], document["encoding"]
        expected_modes = ({"absolute", "racemic"} if source["stereochemistry"] == "RACEMIC"
                          else {"absolute"} if source["stereo_centers"] else set())
        assert set(encodings) == expected_modes
        if "racemic" in encodings:
            absolute, racemic = encodings["absolute"]["inchi"], encodings["racemic"]["inchi"]
            assert racemic.split("/t")[0].replace("InChI=1/", "InChI=1S/") == (
                absolute.split("/t")[0]
            )
            assert re.findall(r"/t[^/]+", racemic) == re.findall(r"/t[^/]+", absolute)
            assert "/s3" in racemic and "/m" not in racemic
        if cas in {"5183-77-7", "57968-71-5"}:
            # RR/SS relative geometry; a meso cis/RS structure has different parity.
            assert "/t5-,6-/s3" in encodings["racemic"]["inchi"]
        if cas == "54638-10-7":
            assert encodings["absolute"]["inchi"].endswith("/t6-/m1/s1")
        if cas == "10395-45-6":
            assert encodings["absolute"]["inchi"].endswith("/t8-/m0/s1")
        if cas == "20089-07-0":
            assert encodings["absolute"]["inchi"].endswith("/t5-/m0/s1")
        if cas == "42569-16-4":
            assert "/b4-3+" in encodings["absolute"]["inchi"]
            assert "/b4-3+" in encodings["racemic"]["inchi"]
    return documents


def source_conflicts(cas, pubchem, srs):
    conflicts = []
    if cas == "55156-16-6":
        record = srs["current"]["identities"]["268737cb-f8bd-4c85-bb40-78d9d72d1a59"]
        structure = record["structure"]
        assert structure["stereochemistry"] == "ACHIRAL" and structure["stereoCenters"] == 1
        conflicts.append({"source": "GSRS current CAS 565-61-7 via SRS former relation",
                          "uuid": record["uuid"], "version": record["version"],
                          "stereochemistry": structure["stereochemistry"],
                          "stereoCenters": structure["stereoCenters"],
                          "opticalActivity": structure["opticalActivity"],
                          "inchi": structure["_inchi"],
                          "decision": "CRC84 explicitly defines the original CAS as racemic; "
                                      "retain the inconsistent GSRS ACHIRAL classification."})
    if cas == "57968-71-5":
        original = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
        annotation = original["candidates"][0]["CAS_matches"][0]["reference"]
        assert annotation["Name"] == "racemic-Dimethoxysuccinic acid, dimethyl ester"
        conflicts.append({"source": "ChemIDplus original CAS annotation",
                          "annotation": annotation,
                          "decision": "CRC84 names and draws diethyl DL-tartrate. "
                                      "The deposited graph agrees in connectivity but its "
                                      "annotation name describes a different constitutional "
                                      "isomer; do not add that name as an alias."})
    return conflicts


def check_identity(component, documents, pubchem, srs, units):
    cas = component.cas_number
    assert documents is not None and cas in documents
    document = documents[cas]
    source, encodings = document["source"], document["encoding"]
    scope, external = component.metadata["identity_scope"], component.metadata["external_identity"]
    assert scope["source_CAS"] == cas and scope["evidence_route"] == ROUTE
    assert scope["source_units"] == units[cas]
    assert component.formula == source["formula"]
    assert component.composition == parse_formula(source["formula"])
    assert component.molecular_weight_g_mol == source["molecular_weight_g_mol"]
    assert component.charge == 0 and source["name"] in component.aliases
    assert scope["stereochemistry"] == source["stereochemistry"]
    assert scope["stereo_centers"] == source["stereo_centers"]
    assert external["CRC84_identity"] == source
    original = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
    assert external["original_PubChem_comparison"] == original
    assert component.metadata["source_conflicts"] == source_conflicts(cas, pubchem, srs)
    provenance = component.provenance[0]
    assert provenance.source_id == "CRC84" and provenance.source_path == URL
    assert provenance.source_key == f"{source['entry']}:{cas}"
    if source["stereochemistry"] == "RACEMIC":
        assert scope["enantiomer_mole_fractions"] == [0.5, 0.5]
        assert scope["absolute_configuration"] == "not_selected"
        assert scope["material_kind"] == "defined_racemate"
        chosen = encodings["racemic"]
        transformation = component.metadata["identity_transformation"]
        assert transformation["representative_InChI"] == encodings["absolute"]["inchi"]
        assert transformation["representative_InChIKey"] == encodings["absolute"]["inchi_key"]
        assert transformation["options"] == chosen["options"]
        if cas == "5183-77-7":
            assert scope["relative_geometry"] == "trans"
        if cas == "57968-71-5":
            assert scope["relative_geometry"] == "RR/SS (DL), not meso"
        if cas == "42569-16-4":
            assert scope["double_bond_geometry"] == "E (CRC84 diagram)"
        assert component.inchi == chosen["inchi"] and component.inchi_key == chosen["inchi_key"]
    elif source["stereochemistry"] == "ABSOLUTE":
        assert scope["material_kind"] == "single_absolute_stereoisomer"
        assert scope["absolute_configuration"] == ("R" if cas == "54638-10-7" else "S")
        assert component.inchi == encodings["absolute"]["inchi"]
        assert component.inchi_key == encodings["absolute"]["inchi_key"]
        assert "enantiomer_mole_fractions" not in scope
    else:
        assert cas == "13454-99-4" and not component.inchi and not component.inchi_key
        assert scope["material_kind"] == "named_stoichiometric_compound"
        assert scope["absolute_configuration"] == "not_specified"
    if original["candidates"]:
        # A generic graph is a connectivity cross-check only, never stereo evidence.
        assert all(c["InChI"].split("/t")[0] == encodings["absolute"]["inchi"].split("/t")[0]
                   for c in original["candidates"])
        if cas == "20089-07-0":
            assert all(c["InChI"] == component.inchi and c["InChIKey"] == component.inchi_key
                       for c in original["candidates"])
    assert len(component.provenance) == (2 if encodings else 1)


def negative_variants(component):
    kind = component.metadata["identity_scope"]["stereochemistry"]
    modes = ["wrong_CAS", "wrong_molecular_weight", "charged_material",
             "wrong_source_entry", "wrong_key"]
    if kind == "RACEMIC":
        modes += ["lost_racemate_scope", "unequal_enantiomers", "absolute_representative",
                  "missing_relative_layer", "relative_not_racemic"]
    elif kind == "ABSOLUTE":
        modes += ["opposite_enantiomer", "wrong_absolute_assignment", "imposed_racemate"]
    else:
        modes += ["invented_InChI", "unknown_mixture"]
    if component.cas_number == "42569-16-4":
        modes += ["missing_double_bond_geometry", "opposite_double_bond_geometry"]
    for mode in modes:
        data = json.loads(json.dumps(component.to_dict()))
        scope = data["metadata"]["identity_scope"]
        if mode == "wrong_CAS":
            data["cas_number"] = "615-38-3"
        elif mode == "wrong_molecular_weight":
            data["molecular_weight_g_mol"] += 0.1
        elif mode == "charged_material":
            data["charge"] = 1
        elif mode == "wrong_source_entry":
            data["provenance"][0]["source_key"] = "1660:13537-23-0"
        elif mode == "wrong_key":
            data["inchi_key"] = "UHHCYAAVGADGGP-UHFFFAOYSA-N"
        elif mode == "lost_racemate_scope":
            scope["stereochemistry"] = "ACHIRAL"
        elif mode == "unequal_enantiomers":
            scope["enantiomer_mole_fractions"] = [0.75, 0.25]
        elif mode == "absolute_representative":
            data["inchi"] = data["metadata"]["identity_transformation"]["representative_InChI"]
        elif mode == "missing_relative_layer":
            data["inchi"] = re.sub(r"/t[^/]+", "", data["inchi"])
        elif mode == "relative_not_racemic":
            data["inchi"] = data["inchi"].replace("/s3", "/s2")
        elif mode == "opposite_enantiomer":
            data["inchi"] = re.sub(r"/m([01])", lambda m: "/m" + str(1 - int(m[1])), data["inchi"])
        elif mode == "wrong_absolute_assignment":
            scope["absolute_configuration"] = "S" if scope["absolute_configuration"] == "R" else "R"
        elif mode == "imposed_racemate":
            scope["stereochemistry"] = "RACEMIC"
        elif mode == "invented_InChI":
            data["inchi"] = "InChI=1S/Cl2F3P/c1-6(2,3,4)5"
        elif mode == "missing_double_bond_geometry":
            data["inchi"] = re.sub(r"/b[^/]+", "", data["inchi"])
        elif mode == "opposite_double_bond_geometry":
            data["inchi"] = data["inchi"].replace("/b4-3+", "/b4-3-")
        else:
            scope["material_kind"] = "mixture_unknown_composition"
        yield mode, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inchi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", choices=("initial", "additional"), default="initial")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started, rows = time.perf_counter(), []
    selected = INITIAL_CAS if args.batch == "initial" else ADDITIONAL_CAS
    total = 22 if args.batch == "initial" else 7
    try:
        sources = read_source(args.source, selected)
        for cas, source in sources.items():
            if not source["stereo_centers"]:
                continue
            mol = args.output / f"{cas}.mol"
            mol.write_text(diagram_mol(cas), encoding="utf8")
            modes = ("absolute", "racemic") if source["stereochemistry"] == "RACEMIC" else (
                "absolute",
            )
            for mode in modes:
                row = {"source_CAS": cas, "mode": mode}
                try:
                    options = ["/Key", "/AuxNone", "/NoLabels"]
                    if mode == "racemic":
                        options.append("/SRac")
                    stem = f"{cas}-{mode}"
                    command = [str(args.inchi), str(mol)] + [
                        str(args.output / f"{stem}.{ext}") for ext in ("txt", "log", "prb")
                    ] + options
                    run = subprocess.run(command, capture_output=True, text=True, timeout=15)
                    row.update(command=command, options=options, returncode=run.returncode)
                    (args.output / f"{stem}.stderr").write_text(run.stderr, encoding="utf8")
                    assert run.returncode == 0 and "1.07.5" in run.stderr
                    raw = (args.output / f"{stem}.txt").read_text(encoding="utf8")
                    (inchi,) = re.findall(r"^InChI=.+$", raw, re.M)
                    (key,) = re.findall(r"^InChIKey=(.+)$", raw, re.M)
                    row.update(inchi=inchi, inchi_key=key)
                    assert inchi.split("/")[1] == source["formula"]
                    (stereo,) = re.findall(r"/t([^/]+)", inchi)
                    assert len(stereo.split(",")) == source["stereo_centers"]
                    assert all(v.endswith(("+", "-")) for v in stereo.split(","))
                    assert "/s3" in inchi and "/m" not in inchi if mode == "racemic" else (
                        "/s1" in inchi and "/m" in inchi
                    )
                    row["status"] = "encoded"
                except Exception:
                    row.update(status="failed", failure=traceback.format_exc())
                rows.append(row)
                print(f"{len(rows)}/{total} {cas} {mode}: {row['status']}", flush=True)
        failure = None
    except Exception:
        failure = traceback.format_exc()
    result = {"task": "NATIVE-CONV-22G1", "mode": "development", "block": "CRC84-encoding",
              "reference_version": "InChI 1.07.5", "total": total, "results": rows,
              "encoded": sum(r["status"] == "encoded" for r in rows),
              "failed": sum(r["status"] == "failed" for r in rows),
              "setup_failure": failure, "seconds": time.perf_counter() - started}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    return result["encoded"] != total


if __name__ == "__main__":
    raise SystemExit(main())
