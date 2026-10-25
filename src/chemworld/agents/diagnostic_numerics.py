"""Bounded, public arithmetic only; STDIO MCP for the final Work II diagnostic."""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class NumericsBudget:
    """One saved policy for calculator admission and follow-up completion."""

    limit: int = 8
    on_exhaustion: str = "terminate_turn"

    def __post_init__(self) -> None:
        if type(self.limit) is not int or not 1 <= self.limit <= 256:
            raise ValueError("numerics limit must be an integer in 1..256")
        if self.on_exhaustion not in {"terminate_turn", "continue_answer"}:
            raise ValueError("unknown numerics exhaustion policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "public-numerics-budget-v1",
            "limit": self.limit,
            "on_exhaustion": self.on_exhaustion,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any] | None) -> NumericsBudget:
        if record is None:
            return cls()  # Historical designs retain their eight-call hard stop.
        if record.get("version") != "public-numerics-budget-v1":
            raise ValueError("unknown saved numerics budget version")
        return cls(limit=record["limit"], on_exhaustion=record["on_exhaustion"])

    def disclosure(self) -> str:
        if self.on_exhaustion == "terminate_turn":
            return (
                f"This follow-up allows {self.limit} calculator attempts, including invalid "
                "expressions. Exceeding that allowance terminates the turn."
            )
        return (
            f"This follow-up has {self.limit} public calculator attempts, including invalid "
            "expressions. Scalar and array calculations share this allowance. Each response "
            "reports attempts used and remaining. After exhaustion, further calculations are "
            "rejected; submit your answer using existing evidence without retrying the tool. "
            "The response deadline still applies."
        )


FOLLOWUP_NUMERICS = NumericsBudget(limit=128, on_exhaustion="continue_answer")

FUNCTIONS: dict[str, Callable[..., Any]] = {
    "array": np.asarray,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "mean": np.mean,
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
    "clip": np.clip,
}
BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _bounded(value: Any) -> Any:
    values = np.asarray(value, dtype=float)
    if values.size > 4096 or not np.isfinite(values).all():
        raise ValueError("nonfinite or oversized numerical result")
    return values.tolist() if values.ndim else float(values)


def _broadcast(*values: Any) -> None:
    shape = np.broadcast_shapes(*(np.shape(value) for value in values))
    if math.prod(shape) > 4096:
        raise ValueError("broadcast size limit")


def calculate(expression: str) -> Any:
    """Evaluate an allowlisted expression; never eval Python, attributes, or names."""
    if not isinstance(expression, str) or len(expression) > 20_000:
        raise ValueError("expression length limit")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 1024:
        raise ValueError("expression node limit")

    def visit(node: ast.AST) -> Any:
        value: Any
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            value = float(cast(float, node.value))
        elif isinstance(node, (ast.List, ast.Tuple)):
            items = [visit(item) for item in node.elts]
            if sum(np.size(item) for item in items) > 4096:
                raise ValueError("array size limit")
            value = np.asarray(items, dtype=float)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            left, right = visit(node.left), visit(node.right)
            _broadcast(left, right)
            value = BINOPS[type(node.op)](left, right)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [visit(item) for item in node.args]
            name = node.func.id
            if name == "linspace" and len(args) == 3:
                if any(np.ndim(item) for item in args):
                    raise ValueError("linspace requires scalar arguments")
                count = float(args[2])
                if not count.is_integer() or not 1 <= count <= 4096:
                    raise ValueError("linspace count limit")
                value = np.linspace(args[0], args[1], int(count))
            elif name == "lstsq" and len(args) == 2:
                a, b = (np.asarray(item, dtype=float) for item in args)
                if a.ndim != 2 or max(a.shape) > 64 or b.ndim != 1 or len(b) != len(a):
                    raise ValueError("lstsq requires <=64x64 design and matched vector")
                value = np.linalg.lstsq(a, b, rcond=None)[0]
            elif name in FUNCTIONS:
                if len(args) != (3 if name == "clip" else 1):
                    raise ValueError("function argument count")
                _broadcast(*args)
                value = FUNCTIONS[name](*args)
            else:
                raise ValueError("unsupported function")
        else:
            raise ValueError("only literal numerical arithmetic and allowlisted functions")
        _bounded(value)
        return value

    with np.errstate(all="raise"):
        return _bounded(visit(tree.body))


def serve(audit: Path, limit: int, *, budget_feedback: bool = False) -> None:
    """Serve JSON-line MCP. All attempts are logged, including rejected expressions."""
    count = len(audit.read_text(encoding="utf-8").splitlines()) if audit.exists() else 0
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            if "id" not in request:
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": request.get("params", {}).get(
                        "protocolVersion", "2024-11-05"
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "public_numerics", "version": "1"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "calculate",
                            "description": (
                                "Public arithmetic only. Literal scalar/array expressions "
                                "with +,-,*,/,**; array,log,exp,sqrt,abs,mean,sum,min,max,clip,"
                                "linspace(start,stop,count),lstsq(design_matrix,target_vector). "
                                "No variables/files/network/simulator. "
                                "For an intercept include a ones column in lstsq. "
                                f"At most {limit} calls."
                                + (
                                    " All attempts count, including invalid expressions. "
                                    "Responses report remaining attempts. Once exhausted, "
                                    "submit your answer without further calculator calls."
                                    if budget_feedback
                                    else ""
                                )
                            ),
                            "inputSchema": {
                                "type": "object",
                                "properties": {"expression": {"type": "string"}},
                                "required": ["expression"],
                                "additionalProperties": False,
                            },
                        }
                    ]
                }
            elif method == "tools/call":
                count += 1
                params = request.get("params", {})
                started = time.perf_counter()
                row = {
                    "attempt": count,
                    "name": params.get("name"),
                    "arguments": params.get("arguments", {}),
                }
                try:
                    if count > limit or params.get("name") != "calculate":
                        raise ValueError("tool attempt limit or unsupported tool")
                    row["value"] = calculate(params.get("arguments", {}).get("expression"))
                    row["status"] = "completed"
                except (
                    ValueError,
                    TypeError,
                    SyntaxError,
                    ArithmeticError,
                    MemoryError,
                    RecursionError,
                ) as error:
                    row.update(status="rejected", error=type(error).__name__)
                    if budget_feedback and count > limit:
                        row.update(
                            error_code="calculation_budget_exhausted",
                            message="No calculator attempts remain. Submit your answer from "
                            "existing evidence; do not retry this tool.",
                        )
                if budget_feedback:
                    row["budget"] = {
                        "limit": limit,
                        "attempts_used": count,
                        "remaining": max(0, limit - count),
                    }
                row["elapsed_s"] = time.perf_counter() - started
                audit.parent.mkdir(parents=True, exist_ok=True)
                with audit.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                result = {
                    "content": [{"type": "text", "text": json.dumps(row)}],
                    "isError": row["status"] != "completed",
                }
            elif method == "ping":
                result = {}
            else:
                raise ValueError("unsupported MCP method")
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except (ValueError, TypeError, KeyError) as error:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": type(error).__name__},
            }
        print(json.dumps(response, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--budget-feedback", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.limit) or not 1 <= args.limit <= 256:
        raise ValueError("tool limit must be 1..256")
    serve(args.audit, args.limit, budget_feedback=args.budget_feedback)


if __name__ == "__main__":
    main()
