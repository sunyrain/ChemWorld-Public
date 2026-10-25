"""Use reviewed CRC identities with the existing native property input API.

Run with a directory containing the external CRC tables, for example:
  uv run --no-sync python examples/crc_electro_optical_inputs.py SOURCE_DIRECTORY
No external property implementation is imported or called by this example.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from chemworld.data.electro_optical import ElectroOpticalDatabase
from chemworld.physchem.electro_optics import parallel_plate_capacitance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec


class SourceIdentityGapError(ValueError):
    """A source row has no reviewed material input in this example."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic['source_CAS']}: {diagnostic['reason']}")


def load_inputs():
    return json.loads(
        (Path(__file__).parent / "fixtures/crc_electro_optical_components.json").read_text(
            encoding="utf8"
        )
    )


def inspect_source_identity(payload, cas, source_unit):
    """Report the reviewed fixture's scope; never infer an alias or composition."""
    for gap in payload.get("source_gaps", ()):
        if gap["source_CAS"] == cas:
            return {**deepcopy(gap), "status": "source_gap", "requested_unit": source_unit}
    for component in payload["components"]:
        if component["cas_number"] == cas:
            units = component["metadata"]["identity_scope"]["source_units"]
            reviewed = source_unit in units
            return {
                "source_CAS": cas,
                "requested_unit": source_unit,
                "source_units": list(units),
                "status": "reviewed" if reviewed else "unreviewed_source_row",
                "reason": "Reviewed material input for this source row"
                if reviewed else "Material review does not cover the requested source row",
            }
    return {
        "source_CAS": cas, "requested_unit": source_unit, "source_units": [],
        "status": "unknown_CAS", "reason": "No reviewed material input for this exact CAS",
    }


def reviewed_property_input(database, payload, identity, *, method, **kwargs):
    """Use the existing input API only for an explicitly reviewed original row.

    This is the example's supplied-input boundary, not an automatic chemical
    identity validator or a restriction on callers' own ComponentSpec objects.
    """
    record = database.record(identity, method=method)
    cas = record.cas_number
    diagnostic = inspect_source_identity(payload, cas, f"{method}/{record.row_number - 1}")
    if diagnostic["status"] != "reviewed":
        raise SourceIdentityGapError(diagnostic)
    component = next(c for c in payload["components"] if c["cas_number"] == cas)
    return database.property_input(
        record.record_id, ComponentSpec.from_dict(component), method=method, **kwargs
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    args = parser.parse_args()
    payload = load_inputs()
    database = ElectroOpticalDatabase(args.source_directory)
    results = []
    for cas in ("590-11-4", "590-12-5"):
        record = database.record(cas, method="CRC_PERMITTIVITY")
        temperature = record.value("T").value
        card = reviewed_property_input(
            database, payload, cas, method="CRC_PERMITTIVITY", source_point=True
        )
        component = card.package.component
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
