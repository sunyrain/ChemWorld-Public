"""Independent final PH closure for all six fixed real-world histories."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from tests.reference.run_incompressible import compare


def main():
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817
    from scipy.optimize import brentq

    if CoolProp.__version__ != "8.0.0":
        raise ValueError("world reference requires CoolProp 8.0.0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite earlier world evidence")
    tree = ET.parse(args.source)
    histories = [
        json.loads(e.attrib["value"])
        for e in tree.findall(".//property[@name='incompressible_history']")
    ]
    events = [
        json.loads(e.attrib["value"])
        for e in tree.findall(".//property[@name='incompressible_event']")
    ]
    lookup = {(r["fluid"], r["area_m2"]): r for r in histories}
    rows = []

    def final(name, area):
        history = lookup[(name, area)]
        if not history.get("completed"):
            raise ValueError("original world history did not complete its fixed checks")
        return history["stages"][-1]

    def independent_temperature(name, area, tank):
        record = final(name, area)
        n = record["amounts_mol"][tank]
        masses = {k: v * record["component_masses_kg_mol"][k] for k, v in n.items()}
        mass = sum(masses.values())
        x = masses[record["solute_id"]] / mass
        ref = CP.AbstractState("INCOMP", name)
        ref.set_mass_fractions([x])

        def residual(T):
            ref.update(CP.PT_INPUTS, record["pressure_Pa"], T)
            return (
                mass * ref.hmass()
                + record["support_capacities_J_K"][tank] * (T - record["support_reference_K"])
                - record["enthalpies_J"][tank]
            )

        return brentq(residual, 290.0, 312.0, xtol=1e-12)

    for name in ("MEG", "MPG", "MCA"):
        for area in (0.0, 0.001):
            for tank in (0, 1):
                rows.append(
                    compare(
                        {"fluid": name, "area_m2": area, "tank": tank},
                        "independent_PH_temperature_K",
                        lambda name=name, area=area, tank=tank: final(name, area)["temperatures_K"][
                            tank
                        ],
                        lambda name=name, area=area, tank=tank: independent_temperature(
                            name, area, tank
                        ),
                        3e-7,
                        0.0,
                    )
                )
    assert len(rows) == 12
    result = {
        "evidence_mode": "development",
        "reference": "CoolProp 8.0.0 with SciPy Brent",
        "source": args.source.name,
        "histories": histories,
        "events": events,
        "comparisons": rows,
        "counts": dict(Counter(r["status"] for r in rows)),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["counts"])


if __name__ == "__main__":
    main()
