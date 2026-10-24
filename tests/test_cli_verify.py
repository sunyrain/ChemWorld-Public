from __future__ import annotations

import json

from chemworld.cli import main
from chemworld.data.records import trajectory_value


def test_cli_run_verify_and_inspect(tmp_path, capsys) -> None:
    trajectory = tmp_path / "run.jsonl"
    manifest = tmp_path / "run.manifest.json"
    main(
        [
            "run",
            "--env",
            "ChemWorld",
            "--agent",
            "random",
            "--budget",
            "8",
            "--seed",
            "5",
            "--output",
            str(trajectory),
            "--manifest",
            str(manifest),
        ]
    )
    main(["verify", "--constitution", "--submission", str(trajectory)])
    main(["inspect-constitution", "--env", "ChemWorld"])
    output = capsys.readouterr().out
    assert "PhysicalConstitutionChecklist" in output

    first_record = json.loads(trajectory.read_text(encoding="utf-8").splitlines()[0])
    assert first_record["operation_type"] == "add_solvent"
    assert "constitution_checks" in first_record
    assert "species_amounts" not in trajectory_value(first_record, 'observation')
    records = [
        json.loads(line)
        for line in trajectory.read_text(encoding="utf-8").splitlines()
    ]
    measure_records = [record for record in records if record["operation_type"] == "measure"]
    assert measure_records
    assert measure_records[-1]["measurement_cost"] > 0
    assert measure_records[-1]["sample_consumed"] > 0
    notes = json.loads(manifest.read_text(encoding="utf-8"))["notes"]
    assert notes["env_id"] == "ChemWorld"


def test_suite_scripted_agent(tmp_path) -> None:
    main(
        [
            "suite",
            "--env",
            "ChemWorld",
            "--agent",
            "scripted_chemistry",
            "--world-splits",
            "public-test",
            "--seeds",
            "0",
            "--budget",
            "12",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "suite_results.json").exists()
