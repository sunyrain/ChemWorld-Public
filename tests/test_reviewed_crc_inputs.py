"""The supplied example rejects unreviewed material without blocking custom inputs."""

import json
import runpy
from pathlib import Path

import pytest

from chemworld.data.electro_optical import ElectroOpticalDatabase
from chemworld.physchem.specs import ComponentSpec

EXAMPLE = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "examples/crc_electro_optical_inputs.py"
))


def test_review_is_bound_to_original_source_rows_and_exact_cas():
    payload = EXAMPLE["load_inputs"]()
    inspect = EXAMPLE["inspect_source_identity"]
    component = payload["components"][0]
    cas = component["cas_number"]
    unit = component["metadata"]["identity_scope"]["source_units"][0]
    assert inspect(payload, cas, unit)["status"] == "reviewed"
    assert inspect(payload, cas, "CRC_RI/999999")["status"] == "unreviewed_source_row"
    assert inspect(payload, component["aliases"][0], unit)["status"] == "unknown_CAS"
    assert inspect(payload, "000-00-0", unit)["status"] == "unknown_CAS"


def test_gap_diagnosis_survives_json_and_cannot_be_overridden_by_representative(tmp_path):
    payload = EXAMPLE["load_inputs"]()
    cas = "96-19-5"
    gap = next(g for g in payload["source_gaps"] if g["source_CAS"] == cas)
    row = int(gap["source_units"][0].split("/")[1])
    directory = tmp_path / "Misc"
    directory.mkdir()
    # Match the original reviewed row address in a disposable external source.
    (directory / "CRC Handbook Organic RI.csv").write_text(
        "CAS\tRI\tRIT\n" + "999-99-9\t1.1\t293.15\n" * row
        + f"{cas}\t1.503\t293.15\n", encoding="utf8"
    )
    database = ElectroOpticalDatabase(tmp_path)
    representative = ComponentSpec("caller-defined", "C3H3Cl3", cas_number=cas)
    payload["components"].append(representative.to_dict())
    payload = json.loads(json.dumps(payload))
    with pytest.raises(EXAMPLE["SourceIdentityGapError"]) as exc:
        EXAMPLE["reviewed_property_input"](
            database, payload, cas, method="CRC_RI", source_point=True
        )
    diagnostic = exc.value.diagnostic
    assert diagnostic["status"] == "source_gap"
    assert diagnostic["category"] == "unspecified_isomer_mixture"
    assert diagnostic["reason"] == gap["reason"]
    assert database.value(cas, "RI", method="CRC_RI").value == 1.503
    # The native API remains caller-programmable; it does not claim to verify chemistry.
    card = database.property_input(cas, representative, method="CRC_RI", source_point=True)
    assert card.package.component == representative
