"""Configure, measure and resume an explicitly declared salt-gradient generator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.physchem.reverse_electrodialysis import ReverseElectrodialysisSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.reverse_electrodialysis import read_reverse_electrodialysis_electrical
from chemworld.world.reverse_electrodialysis_world import reverse_electrodialysis_world_spec


def example_report(
    spec: ReverseElectrodialysisSpec,
    amounts: tuple[Mapping[str, float], Mapping[str, float]],
    *,
    shared_clock: bool = True,
) -> dict[str, Any]:
    composition = reverse_electrodialysis_world_spec(spec, *amounts, shared_clock=shared_clock)
    runtime, restored = ChemWorldRuntime(composition=composition), None
    actions = [
        {"operation": "configure_electrical_control", "control_mode": "open_circuit"},
        {
            "operation": "configure_electrical_control",
            "control_mode": "resistance",
            "resistance_Ohm": 100,
        },
        {"operation": "wait", "duration_s": 1},
        {"operation": "measure", "instrument": "electrical_meter"},
        {
            "operation": "configure_electrical_control",
            "control_mode": "resistance",
            "resistance_Ohm": 200,
            "ramp_duration_s": 1,
        },
    ]
    try:
        runtime.reset(seed=48)
        initial = runtime._state.to_dict()
        transactions = []
        for action in actions:
            result = runtime.execute(action)
            transactions.append({"action": action, "info": result.info})
            if result.info["transaction_status"] != "committed":
                return {
                    "transactions": transactions,
                    "initial_state": initial,
                    "final_state": runtime._state.to_dict(),
                    "complete": False,
                }
        prefix = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
        terminal = read_reverse_electrodialysis_electrical(runtime._state)
        restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(prefix)))
        tail = {"operation": "wait", "duration_s": 2}
        original_result, replay_result = runtime.execute(tail), restored.execute(tail)
        return {
            "complete": True,
            "transactions": transactions,
            "tail_transactions": [original_result.info, replay_result.info],
            "prefix_snapshot": prefix,
            "tail_action": tail,
            "composition": composition.to_dict(),
            "initial_state": initial,
            "final_state": runtime._state.to_dict(),
            "final_snapshot": json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False)),
            "replayed_state": restored._state.to_dict(),
            "exact_json_resume": runtime.snapshot().to_dict() == restored.snapshot().to_dict(),
            "prefix_terminal": {
                "voltage_V": terminal.terminal_potential_V,
                "current_A": terminal.applied_current_A,
            },
        }
    finally:
        runtime.close()
        if restored:
            restored.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declaration",
        type=Path,
        required=True,
        help="JSON object with model, diluate and concentrate declarations",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.declaration.read_text(encoding="utf-8"))
    result = example_report(
        ReverseElectrodialysisSpec.from_dict(data["model"]), (data["diluate"], data["concentrate"])
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"complete": result["complete"], "exact_json_resume": result.get("exact_json_resume")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
