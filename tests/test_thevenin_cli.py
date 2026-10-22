"""Original render/verify CLI consumes the circuit's declared current native world."""

import io
import json
from contextlib import redirect_stdout

import pytest

from chemworld.cli import main
from chemworld.data.logging import load_jsonl
from chemworld.eval.verify import verify_records
from chemworld.examples.native_thevenin import workflow


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_rendered_circuit_trajectory_has_all_original_records_and_replays(tmp_path, noise):
    world, actions = workflow()
    composition = tmp_path / "world.json"
    composition.write_text(json.dumps(world.to_dict(), sort_keys=True), encoding="utf8")
    paths = []
    for index, action in enumerate(actions):
        path = tmp_path / f"action-{index}.json"
        path.write_text(json.dumps(action), encoding="utf8")
        paths.append(str(path))
    trajectory = tmp_path / "cli.jsonl"
    with redirect_stdout(io.StringIO()):
        main(["render", "--composition", str(composition), "--seed", "7",
              "--noise-mode", noise, "--observation-seed", "12", "--actions", *paths,
              "--format", "json", "--trajectory", str(trajectory)])
    records = load_jsonl(trajectory)
    assert len(records) == 9 and all(r["transaction_status"] == "committed" for r in records)
    report = verify_records(records, tolerance=1e-8)
    assert report.verified and report.checked_steps == 9
