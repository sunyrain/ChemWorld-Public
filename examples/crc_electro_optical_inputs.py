"""Use reviewed CRC identities with the existing native property input API.

Run with a directory containing the external CRC tables, for example:
  uv run --no-sync python examples/crc_electro_optical_inputs.py SOURCE_DIRECTORY
No external property implementation is imported or called by this example.
"""

import argparse
import json
from pathlib import Path

from chemworld.data.electro_optical import ElectroOpticalDatabase
from chemworld.physchem.electro_optics import parallel_plate_capacitance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    args = parser.parse_args()
    payload = json.loads(
        (Path(__file__).parent / "fixtures/crc_electro_optical_components.json").read_text(
            encoding="utf8"
        )
    )
    components = {c["cas_number"]: ComponentSpec.from_dict(c) for c in payload["components"]}
    database = ElectroOpticalDatabase(args.source_directory)
    results = []
    for cas in ("590-11-4", "590-12-5"):
        component = components[cas]
        record = database.record(cas, method="CRC_PERMITTIVITY")
        temperature = record.value("T").value
        card = database.property_input(cas, component, method="CRC_PERMITTIVITY", source_point=True)
        card = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
        report = card.package.evaluate("relative_permittivity", temperature_K=temperature)
        # Fixed example geometry, not a measurement of an actual device.
        capacitance = parallel_plate_capacitance(report, area_m2=0.01, separation_m=0.001)
        results.append(
            {
                "CAS": cas,
                "InChI": component.inchi,
                "temperature_K": temperature,
                "relative_permittivity": report.value,
                "capacitance_F": capacitance.value,
                "identity_scope": component.metadata["identity_scope"],
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
