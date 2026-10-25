"""Publish every installed SDE attempt without duplicating per-module absolute paths."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="conv51e-package-*")
    parser.add_argument("--schema", default="chemworld-sde51e-package-history-v1")
    args = parser.parse_args()
    attempts = []
    unfinished = []
    for directory in sorted(args.evidence.glob(args.pattern)):
        if not directory.is_dir():
            continue
        source = directory / "summary.json"
        if not source.is_file():
            unfinished.append(str(directory))
            continue
        result = json.loads(source.read_text(encoding="utf-8"))
        probe = result.get("protocol_summary")
        if probe is not None:
            origins = probe.pop("module_origins")
            target = (directory / "installed").resolve()
            probe["modules_loaded"] = len(origins)
            probe["all_modules_from_installed"] = all(
                Path(origin).is_relative_to(target) for origin in origins.values()
            )
            if not probe["all_modules_from_installed"]:
                raise ValueError("installed module source mismatch")
        if args.pattern.startswith("conv51f-package-"):
            protocols = result.get("protocol_summary", {}).get("protocols", [])
            result.setdefault("protocols_planned", 8)
            result.setdefault("protocols_recorded", len(protocols))
            result.setdefault("protocols_passed", sum(p["passed"] for p in protocols))
            result.setdefault("protocols_without_final_record", 8 - len(protocols))
        attempts.append({"attempt": directory.name, "raw_evidence": str(directory), **result})
    args.output.write_text(
        json.dumps(
            {"schema": args.schema, "attempts": attempts, "unfinished_attempts": unfinished},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"attempts": len(attempts), "passed": sum(a["passed"] for a in attempts)}))


if __name__ == "__main__":
    main()
