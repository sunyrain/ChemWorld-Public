"""Fresh interpreter continuation of one original catalytic world."""

import json
import sys
from pathlib import Path

from chemworld.runtime import ChemWorldRuntime

source, target = sys.argv[1:]
runtime = ChemWorldRuntime.from_snapshot(json.loads(Path(source).read_text(encoding="utf-8")))
for action in (
    {"operation": "wait", "duration_s": 15},
    {"operation": "measure", "instrument": "electrical_meter"},
    {"operation": "measure", "instrument": "chemostatted_coverage"},
):
    result = runtime.execute(action)
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
Path(target).write_text(json.dumps(runtime.snapshot().to_dict(), indent=2), encoding="utf-8")
