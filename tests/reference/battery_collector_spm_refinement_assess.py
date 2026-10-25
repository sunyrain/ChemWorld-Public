"""Incremental, fail-closed assessment of the fixed collector refinement block.

Every trace passes through the complete unchanged supplementary validator.
Only copied comparison arrays survive ingestion; raw histories belong in the
caller's verified archives. This module neither integrates nor writes files.
"""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
from battery_collector_spm_refinement_cases import (
    EDGES,
    FINE_EDGES,
    IMPLEMENTATIONS,
    LAYOUT,
    MESHES,
    history_plan,
    unit_data,
)
from battery_collector_spm_supplement_assess import (
    Checks,
    array,
    canonical,
    compare_histories,
    sample_indices,
    validate_trace,
)


def _compact(normalized):
    """Do not keep views into full radial arrays or any raw trace dictionary."""
    if not normalized:
        raise ValueError("a nonempty validated history is required")
    return [
        {
            "times": np.array(stage["times"], copy=True),
            "values": np.array(stage["values"], copy=True),
            "grid": dict(stage["grid"]),
            "profiles": {
                electrode: {
                    key: np.array(stage["profiles"][electrode][key], copy=True)
                    for key in ("total", "q")
                }
                for electrode in ("negative", "positive")
            },
        }
        for stage in normalized
    ]


class RefinementAssessment:
    """Consume each fixed side once, then evaluate all seventeen comparisons."""

    def __init__(self, unit):
        self._data = unit_data(unit)
        self._plan = history_plan(unit)
        self._book = Checks()
        self._normalized = {}
        self._histories = {}
        self._attempts = 0
        self._finished = None

    @property
    def data(self):
        return deepcopy(self._data)

    def add_history(self, mesh, implementation, trace):
        if self._finished is not None:
            raise RuntimeError("assessment is already finished")
        self._attempts += 1
        request = f"ingestion:{self._attempts}"
        valid = (
            isinstance(mesh, str)
            and isinstance(implementation, str)
            and (mesh, implementation) in self._plan
        )
        self._book.truth(request + ":authored-history", valid)
        if not valid:
            return {"accepted": False, "reason": "unknown mesh or implementation"}
        key = (mesh, implementation)
        unique = key not in self._histories
        self._book.truth(request + ":unique-history", unique)
        if not unique:
            return {"accepted": False, "reason": "duplicate history"}

        label = f"history:{mesh}:{implementation}"
        start = len(self._book.items)
        self._book.truth(
            label + ":execution-success",
            isinstance(trace, dict) and trace.get("exception") is None,
        )
        normalized = self._book.guarded(
            label + ":trace-schema",
            validate_trace,
            self._book,
            label,
            trace,
            self._data,
            MESHES[mesh],
            LAYOUT,
            implementation == "source",
            False,
        )
        compact = (
            self._book.guarded(label + ":comparison-storage", _compact, normalized)
            if normalized is not None
            else None
        )
        self._normalized[key] = compact
        checks = self._book.items[start:]
        record = {
            "mesh": mesh,
            "grid": dict(MESHES[mesh]),
            "implementation": implementation,
            "provided": True,
            "schema_readable": compact is not None,
            "passed": compact is not None and bool(checks) and all(c["passed"] for c in checks),
            "stages": len(compact) if compact is not None else 0,
            "samples": sum(len(s["times"]) for s in compact) if compact is not None else 0,
            "check_range": [start, len(self._book.items)],
        }
        self._histories[key] = record
        return deepcopy(record)

    def _compare(self, name, kind, left, right, *, edge=None):
        start = len(self._book.items)
        self._book.guarded(
            name + ":comparison-schema",
            compare_histories,
            self._book,
            name,
            self._normalized.get(left),
            self._normalized.get(right),
            self._data,
        )
        checks = self._book.items[start:]
        stages = sum(c["name"].endswith(":full-common-coverage") for c in checks)
        self._book.truth(
            name + ":complete-comparison-stages", stages == len(self._data["waveform"])
        )
        checks = self._book.items[start:]
        return {
            "name": name,
            "label": name,
            "kind": kind,
            "left": list(left),
            "right": list(right),
            "edge": list(edge) if edge is not None else None,
            "descriptive_fine_edge": edge in FINE_EDGES if edge is not None else False,
            "stage_denominator": len(self._data["waveform"]),
            "stages": len(self._data["waveform"]),
            "stages_evaluated": stages,
            "passed": bool(checks) and all(c["passed"] for c in checks),
            "check_range": [start, len(self._book.items)],
            "norms": [
                {k: c[k] for k in ("name", "max_abs", "limit", "threshold", "count")}
                for c in checks
                if "threshold" in c
            ],
        }

    def _ratios(self):
        """Descriptive A/B/C differences on identical time/quantity supports.

        The q ratio uses the SAME sixteen intervals at all three levels, unlike
        the separate qualification comparisons whose common partitions differ.
        There is no convergence-order claim or acceptance threshold here.
        """
        reports = []
        for implementation in IMPLEMENTATIONS:
            histories = [self._normalized.get((mesh, implementation)) for mesh in ("A", "B", "C")]
            if any(h is None for h in histories):
                reports.append(
                    {
                        "implementation": implementation,
                        "available": False,
                        "reason": "A/B/C complete histories unavailable",
                    }
                )
                continue
            start = 0.0
            for index, (duration, _target, _mode) in enumerate(self._data["waveform"]):
                authored = np.linspace(start, start + duration, self._data["samples_per_segment"])
                fields = []
                for history in histories:
                    stage = history[index]
                    rows = sample_indices(stage["times"], authored)
                    values = array(stage["values"])[rows]
                    scalars = {
                        name: values[:, column]
                        for column, name in enumerate(
                            ("voltage_V", "current_A", "charge_C", "work_J")
                        )
                    }
                    scalars.update(
                        {
                            name + "-Li": stage["profiles"][name]["total"][rows]
                            for name in ("negative", "positive")
                        }
                    )
                    q = stage["profiles"]["negative"]["q"][rows]
                    scalars["q-16-common-intervals_A"] = q.reshape(len(rows), 16, -1).sum(axis=2)
                    fields.append(scalars)
                for quantity in fields[0]:
                    values = [array(f[quantity]) for f in fields]
                    first = float(np.max(abs(values[0] - values[1])))
                    second = float(np.max(abs(values[1] - values[2])))
                    scale = max(*(float(np.max(abs(v))) for v in values), 1e-300)
                    floor = 64 * np.finfo(float).eps * scale
                    meaningful = min(first, second) > floor
                    ratio = first / second if meaningful else None
                    reason = None if meaningful else "zero or representation-scale difference"
                    if ratio is not None and not np.isfinite(ratio):
                        ratio, reason = None, "nonfinite ratio"
                    reports.append(
                        {
                            "implementation": implementation,
                            "stage": index,
                            "quantity": quantity,
                            "available": True,
                            "numerator_edge": ["A", "B"],
                            "denominator_edge": ["B", "C"],
                            "numerator_max_abs": first,
                            "denominator_max_abs": second,
                            "ratio": ratio,
                            "unavailable_reason": reason,
                            "representation_floor": float(floor),
                            "asymptotic_order": None,
                            "scope": "descriptive only; no acceptance effect or inferred order",
                        }
                    )
                start += duration
        return reports

    def finish(self):
        if self._finished is not None:
            return deepcopy(self._finished)
        self._book.equal("history-denominator", len(self._histories), len(self._plan))
        self._book.equal("history-attempt-denominator", self._attempts, len(self._plan))
        for mesh, implementation in self._plan:
            self._book.truth(
                f"history:{mesh}:{implementation}:present",
                (mesh, implementation) in self._histories,
            )
        comparisons = [
            self._compare(
                f"same-physical-port:{mesh}",
                "same-physical-port",
                (mesh, "native"),
                (mesh, "source"),
            )
            for mesh in MESHES
        ]
        for implementation in IMPLEMENTATIONS:
            for left, right in EDGES:
                comparisons.append(
                    self._compare(
                        f"grid:{implementation}:{left}-{right}",
                        "mesh-refinement",
                        (left, implementation),
                        (right, implementation),
                        edge=(left, right),
                    )
                )
        comparison_count = len(MESHES) + len(EDGES) * len(IMPLEMENTATIONS)
        stage_count = comparison_count * len(self._data["waveform"])
        self._book.equal("comparison-denominator", len(comparisons), comparison_count)
        self._book.equal(
            "comparison-stage-denominator",
            sum(c["stages_evaluated"] for c in comparisons),
            stage_count,
        )
        ratios = self._book.guarded("refinement-ratio-schema", self._ratios)
        fine = [c for c in comparisons if c["descriptive_fine_edge"]]
        failures = [c for c in self._book.items if not c["passed"]]
        report = {
            "id": self._data["id"],
            "inputs": self.data,
            "layout": dict(LAYOUT),
            "meshes": deepcopy(MESHES),
            "edges": list(EDGES),
            "fine_edges": list(FINE_EDGES),
            "history_denominator": len(self._plan),
            "history_attempts": self._attempts,
            "history_count": len(self._histories),
            "histories_provided": len(self._histories),
            "histories_schema_readable": sum(
                h["schema_readable"] for h in self._histories.values()
            ),
            "histories": [
                self._histories.get(
                    (mesh, impl),
                    {
                        "mesh": mesh,
                        "grid": dict(MESHES[mesh]),
                        "implementation": impl,
                        "provided": False,
                        "schema_readable": False,
                        "passed": False,
                        "stages": 0,
                        "samples": 0,
                    },
                )
                for mesh, impl in self._plan
            ],
            "comparison_denominator": comparison_count,
            "comparison_stage_denominator": stage_count,
            "comparisons": comparisons,
            "refinement_ratios": ratios,
            "fine_edge_description": {
                "comparisons": len(fine),
                "passed": sum(c["passed"] for c in fine),
                "strict_acceptance_override": False,
            },
            "checks": self._book.items,
            "failures": failures,
            "passed": bool(self._book.items) and not failures,
        }
        self._finished = json.loads(canonical(report))
        return deepcopy(self._finished)
