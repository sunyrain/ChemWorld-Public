"""Predeclared actual-world 31D canary; all attempts are retained outside Git."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Event, Thread

from chemworld.examples.gas_shaft_machine import gas_shaft_machine_world
from chemworld.runtime import ChemWorldRuntime


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    if result.info["transaction_status"] != "committed":
        raise ValueError(json.dumps(result.info))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    start, done = time.monotonic(), Event()

    def heartbeat():
        while not done.wait(30):
            print(
                f"31D canary completed={len(rows)}/3 elapsed={time.monotonic() - start:.1f}s",
                flush=True,
            )

    Thread(target=heartbeat, daemon=True).start()
    for mode, heating in (("compressor", False), ("expander", False), ("compressor", True)):
        identity = mode + "-heat" if heating else mode
        row = {"id": identity, "checks": {}, "failures": []}
        session = replay = None
        print("31D canary start", identity, flush=True)
        try:
            composition = gas_shaft_machine_world(mode, with_heater=heating)
            session = ChemWorldRuntime(composition=composition)
            session.reset(seed=31)
            if heating:
                execute(session, "select_vessel", vessel="source")
                execute(session, "queue_lab_heat", target_temperature_K=600.0, duration_s=1.0)
                execute(session, "select_vessel", vessel="receiver")
            result = execute(session, "wait", duration_s=1.0)
            row["wait_info"] = result.info
            first = session._state
            row["checks"].update(
                one_clock=first.ledger.time_s == 1.0,
                shaft_sign=first.ledger.flow_shaft_work_J > 0
                if mode == "compressor"
                else first.ledger.flow_shaft_work_J < 0,
                no_piston_work=first.ledger.boundary_work_J == 0.0,
                separate_heat=abs(first.ledger.energy_jacket_J - (2.0 if heating else 0.0)) <= 1e-8,
            )
            payload = json.loads(json.dumps(session.snapshot().to_dict(), allow_nan=False))
            replay = ChemWorldRuntime.from_snapshot(payload)
            source = args.output / f"{identity}-snapshot.json"
            target = args.output / f"{identity}-fresh-state.json"
            source.write_text(json.dumps(payload), encoding="utf8")
            code = (
                "import json; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                f"p=json.loads(Path({str(source)!r}).read_text(encoding='utf8')); "
                "r=ChemWorldRuntime.from_snapshot(p); "
                "z=r.execute({'operation':'wait','duration_s':1.}); "
                "assert z.info['transaction_status']=='committed',z.info; "
                f"p=Path({str(target)!r}); "
                "p.write_text(json.dumps(r._state.to_dict()),encoding='utf8'); r.close()"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            for runtime in (session, replay):
                execute(runtime, "wait", duration_s=1.0)
            row["checks"]["snapshot_equal"] = session._state.to_dict() == replay._state.to_dict()
            row["checks"]["fresh_process_equal"] = json.loads(
                target.read_text(encoding="utf8")
            ) == json.loads(json.dumps(session._state.to_dict()))
            row["final_state"] = session._state.to_dict()
        except Exception as exc:
            row["failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        finally:
            if session:
                session.close()
            if replay:
                replay.close()
        rows.append(row)
        print(identity, row["checks"], "failure_count", len(row["failures"]), flush=True)
        (args.output / "results.json").write_text(
            json.dumps(
                {"rows": rows, "elapsed_s": time.monotonic() - start}, indent=2, allow_nan=False
            ),
            encoding="utf8",
        )
    done.set()


if __name__ == "__main__":
    main()
