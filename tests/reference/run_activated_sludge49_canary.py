"""Actual installed original-world consumer canaries before the full block."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.activated_sludge_world import activated_sludge_world
from reference.activated_sludge49_fixtures import inventory_case, spec_case


def commit(rt, operation, **data):
    result = rt.execute({"operation": operation, **data})
    if result.info["transaction_status"] != "committed":
        raise AssertionError(
            {
                k: result.info.get(k)
                for k in (
                    "transaction_status",
                    "error_message",
                    "rollback_reason",
                    "outcome_diagnostics",
                )
            }
        )
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for n in (1, 5):
        row = {"name": f"installed-{n}-cstr"}
        try:
            spec = spec_case(n)
            world = activated_sludge_world(spec, inventory_case(spec))
            rt = ChemWorldRuntime(composition=world)
            rt.reset(seed=49)
            commit(rt, "set_flow_rate", flow_rate_mL_min=0.1281)
            commit(rt, "wait", duration_s=10.0)
            commit(rt, "select_vessel", vessel="effluent")
            commit(rt, "wait", duration_s=5.0)
            snapshot = json.loads(json.dumps(rt.snapshot().to_dict()))
            clone = ChemWorldRuntime.from_snapshot(snapshot)
            commit(rt, "wait", duration_s=3.0)
            commit(clone, "wait", duration_s=3.0)
            actual = json.loads(json.dumps(rt.snapshot().to_dict()))
            if actual != json.loads(json.dumps(clone.snapshot().to_dict())):
                raise AssertionError("full JSON continuation differs")
            source = output / f"snapshot-{n}.json"
            source.write_text(json.dumps(snapshot), encoding="utf8")
            target = output / f"fresh-{n}.json"
            code = (
                "import json,sys; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                "rt=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text())); "
                "r=rt.execute({'operation':'wait','duration_s':3.}); "
                "assert r.info['transaction_status']=='committed',r.info; "
                "Path(sys.argv[2]).write_text(json.dumps(rt.snapshot().to_dict()))"
            )
            subprocess.run(
                [sys.executable, "-c", code, str(source), str(target)],
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if json.loads(target.read_text()) != actual:
                raise AssertionError("fresh-process complete JSON continuation differs")
            row.update(passed=True, checks=6)
        except Exception as exc:
            row.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
        records.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "traceback"}), flush=True)
    (output / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf8")
    return all(row["passed"] for row in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
