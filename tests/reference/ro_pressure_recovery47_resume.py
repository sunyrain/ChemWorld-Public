"""Original session fresh-process continuation, including signed shaft output."""

import json
import sys
from pathlib import Path

from chemworld.runtime import ChemWorldRuntime


def main():
    runtime = ChemWorldRuntime.from_snapshot(
        json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    )
    result = runtime.execute({"operation": "wait", "duration_s": float(sys.argv[3])})
    Path(sys.argv[2]).write_text(
        json.dumps({"info": result.info, "snapshot": runtime.snapshot().to_dict()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
