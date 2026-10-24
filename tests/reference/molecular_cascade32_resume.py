"""Fresh-process original snapshot continuation, with no alternative executor."""

import json
import sys
from pathlib import Path

from chemworld.runtime import ChemWorldRuntime


def main():
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    runtime = ChemWorldRuntime.from_snapshot(payload)
    result = runtime.execute({"operation": "wait", "duration_s": float(sys.argv[3])})
    Path(sys.argv[2]).write_text(
        json.dumps(
            {
                "snapshot": runtime.snapshot().to_dict(),
                "status": result.info["transaction_status"],
                "outcome": result.info["operation_outcome"],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
