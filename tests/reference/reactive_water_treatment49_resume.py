"""Continue a committed 49B original world snapshot in a fresh interpreter."""

import json
import sys
from pathlib import Path

from chemworld.runtime import ChemWorldRuntime


def main():
    source, destination, operation, duration = sys.argv[1:]
    runtime = ChemWorldRuntime.from_snapshot(json.loads(Path(source).read_text(encoding="utf-8")))
    result = runtime.execute({"operation": operation, "duration_s": float(duration)})
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
    Path(destination).write_text(
        json.dumps(runtime.snapshot().to_dict(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
