"""Independent scalar U inversions of all fixed humid material world histories."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scipy.optimize import brentq
from tests.reference.run_incompressible import compare

from chemworld.physchem.humid_air_inventory import humid_air_volume


def reference_temperature(record):
    declaration, n = record["thermodynamics"], record["amounts_mol"]
    species = [declaration[k] for k in ("nitrogen", "oxygen", "argon", "water")]
    masses = [n[c["identifier"]] * c["molecular_weight_g_mol"] / 1000 for c in species]
    phase = declaration["condensed_phase"]
    low, high = (270.0, 460.0) if phase == "liquid" else (193.0, 272.0)

    def energy(T):
        fluid = humid_air_volume(
            T, sum(masses[:3]), masses[3], record["volume_L"] * 1e-3, condensed_phase=phase
        )
        return (
            fluid.internal_energy_J
            + declaration["support_heat_capacity_J_K"]
            * (T - declaration["reference_temperature_K"])
            - record["energy_J"]
        )

    return brentq(energy, low, high, xtol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite humid world closure evidence")
    histories = [
        json.loads(e.attrib["value"])
        for e in ET.parse(args.source).findall(".//property[@name='humid_air_material_history']")
    ]
    lookup = {(h["phase"], h["mode"]): h for h in histories}
    rows = []
    for phase in ("liquid", "ice"):
        for mode in ("mixing", "dewatering"):
            keys = [("final", "reactor"), ("final", "receiver")]
            if mode == "dewatering":
                keys += [("cooled", "reactor"), ("packet", "reactor")]
            for label, vessel in keys:

                def record(phase=phase, mode=mode, label=label, vessel=vessel):
                    history = lookup[phase, mode]
                    if not history.get("completed"):
                        raise ValueError("fixed history did not complete")
                    return next(s for s in history["stages"] if s["label"] == label)["vessels"][
                        vessel
                    ]

                rows.append(
                    compare(
                        {"phase": phase, "mode": mode, "stage": label, "vessel": vessel},
                        "independent_U_temperature_K",
                        lambda record=record: record()["temperature_K"],
                        lambda record=record: reference_temperature(record()),
                        3e-6,
                        0.0,
                    )
                )
                print(f"independent humid material U closure {len(rows)}/12", flush=True)
    assert len(rows) == 12
    result = {
        "evidence_mode": "development",
        "reference": "SciPy Brent; original G8 potential",
        "reference_scope": "independent inversion, not an independent physical model",
        "source": args.source.name,
        "histories": histories,
        "comparisons": rows,
        "counts": dict(Counter(r["status"] for r in rows)),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["counts"])


if __name__ == "__main__":
    main()
