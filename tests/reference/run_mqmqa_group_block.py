"""Fixed 96 complete, self-authored DAT inputs for additional-ion group semantics.

The isolated reference uses actual pycalphad 0.11.2, not a copied equation.
Inputs and full responses remain outside Git; existing inputs must match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path


def cases():
    for family, lattice, code, groups, extra in product(
            ("SUBG", "SUBQ"), ("cation", "anion"), ("G", "Q"),
            ("111", "121", "122", "123"), (1, 2, 3)):
        name = f"{family}-{lattice}-{code}-{groups}-{extra}.dat"
        elements = ("NA", "K", "LI", "F") if lattice == "cation" else ("NA", "F", "CL", "BR")
        masses = "22.98976928 39.0983 6.94 18.998403163" if lattice == "cation" else (
            "22.98976928 18.998403163 35.45 79.904")
        pairs = [(i, 3) for i in range(3)] if lattice == "cation" else [(0, i) for i in range(1, 4)]
        lines = ["Synthetic additional-ion group response "+name, "4 2 0 6 0",
                 " ".join(elements), masses, "6 1 2 3 4 5 6", "1 1", "MIX", family]
        if family == "SUBG":
            lines.append("2.4")
        lines.append("3 3")
        for c, a in pairs:
            lines += [elements[c]+elements[a],
                      "1 1 "+" ".join(str(int(i in {c, a})) for i in range(4)),
                      "1800 0 0 0 0 0 0", "1 1 0 0 0"]
            if family == "SUBQ":
                lines.append("2.4")
        if lattice == "cation":
            lines += ["3 1", "NA K LI", "F", "1 1 1", " ".join(groups),
                      "1", "1", "1 2 3", "1 1 1"]
            quads = [(i, i, 4, 4) for i in (1, 2, 3)]
            indices, exponents, additional = "1 2 4 4", "1 2 3 0", f"{extra} 0"
        else:
            lines += ["1 3", "NA", "F CL BR", "1", "1", "1 1 1",
                      " ".join(groups), "1 1 1", "1 2 3"]
            quads = [(1, 1, i, i) for i in (2, 3, 4)]
            indices, exponents, additional = "1 1 2 3", "1 2 0 3", f"0 {extra+1}"
        for quad, z in zip(quads, (3, 4, 5), strict=True):
            lines.append(" ".join(map(str, (*quad, z, z, z, z))))
        lines += [f"3 {code} {indices} {exponents}", " ".join(["0"]*12),
                  additional+" 1000", "0"]
        yield name, "\n".join(lines)+"\n"


def prepare_inputs(directory):
    if directory.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("generated scientific inputs must remain outside the checkout")
    authored = dict(cases())
    if directory.exists():
        if {p.name for p in directory.glob("*.dat")} != authored.keys():
            raise ValueError("existing block does not have exactly the fixed 96 inputs")
        for name, text in authored.items():
            if (directory/name).read_bytes() != text.encode("utf-8"):
                raise ValueError("refusing to alter existing scientific input "+name)
        return
    directory.mkdir(parents=True, exist_ok=False)
    for name, text in authored.items():
        (directory/name).write_bytes(text.encode("utf-8"))
    (directory/"inputs.json").write_text(json.dumps({
        "data_kind": "synthetic", "license": "CC0-1.0", "expected_inputs": 96,
        "expected_points": 576, "sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in authored.items()}}, indent=2, sort_keys=True), encoding="utf-8")


def native_source(name):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(DataSourceProvenance("synthetic-mq-group-"+name, 0,
        "Self-authored fixed additional-ion group qualification", "1", "CC0-1.0"), "synthetic")


def main():
    if "--stage" not in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--source-directory", type=Path, required=True)
        args, _ = parser.parse_known_args()
        prepare_inputs(args.source_directory)
    spec = importlib.util.spec_from_file_location("mq_response_driver",
        Path(__file__).with_name("run_mqmqa_block.py"))
    mq_driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq_driver)
    driver = mq_driver.configure_driver(__file__)
    driver.FILES = tuple(name for name, _ in cases())
    driver.PHASE_COUNT, driver.POINT_COUNT = 96, 576
    driver.native_source = native_source
    return driver.main()


if __name__ == "__main__":
    raise SystemExit(main())
