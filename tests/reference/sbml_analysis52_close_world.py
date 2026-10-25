"""Fresh-process actual-world query and continuation for the two new analyses."""

import json
import sys

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_lyapunov import analyze_sbml_lyapunov_world
from chemworld.runtime.sbml_timescales import analyze_sbml_timescales_world


def exercise(runtime, kind):
    before = runtime.snapshot().to_dict()
    composition = runtime.compiled_composition.spec
    if kind == "qr":
        report = analyze_sbml_lyapunov_world(
            runtime._state,
            composition,
            amount_scales_mol={"A": 1, "B": 1},
            qr_interval_s=0.2,
            measurement_duration_s=1.0,
            transient_duration_s=0.4,
        )
    else:
        report = analyze_sbml_timescales_world(
            runtime._state,
            composition,
            amount_scales_mol={"A": 1, "B": 1},
            sample_offsets_s=(0, 0.2, 0.5),
        )
    unchanged = runtime.snapshot().to_dict() == before
    result = runtime.execute({"operation": "wait", "duration_s": 0.3})
    return {
        "report": report,
        "full_snapshot_unchanged": unchanged,
        "followup_committed": result.info["transaction_status"] == "committed",
        "after": runtime.snapshot().to_dict(),
    }


if __name__ == "__main__":
    request = json.load(sys.stdin)
    runtime = ChemWorldRuntime.from_snapshot(request["snapshot"])
    try:
        json.dump(exercise(runtime, request["kind"]), sys.stdout, allow_nan=False)
    finally:
        runtime.close()
