"""Read an explicit retained DFN input block, never silently select a newer run."""

import json


def load_dfn_source(path, attempt_index=None):
    source = json.loads(path.read_text())
    if "attempts" in source:
        if (not isinstance(attempt_index, int) or isinstance(attempt_index, bool)
                or not 0 <= attempt_index < len(source["attempts"])):
            raise ValueError("aggregate input requires an explicit valid zero-based source index")
        source = source["attempts"][attempt_index]
    elif attempt_index is not None:
        raise ValueError("source index applies only to an aggregate input")
    units = source.get("units", [])
    if len(units) != 15 or {r.get("unit") for r in units} != {
        f"D{i:02}" for i in range(1, 16)
    }:
        raise ValueError("requires one complete retained 15-unit source without duplicates")
    return source
