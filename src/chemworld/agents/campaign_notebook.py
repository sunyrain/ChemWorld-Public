"""Optional versioned notebook alongside the existing public campaign MCP tools.

This adapter leaves physical execution and sealed posttests to their original runners.
The notebook process only reads the host-published operation stream, never trajectories,
private world parameters, provider credentials, or evaluator answers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from chemworld.agents.experiment_codex_mcp import ChemWorldMCPServer
from chemworld.agents.experiment_documents import ExperimentDocumentWorkspace
from chemworld.agents.hypothesis_notebook import HYPOTHESIS_NOTEBOOK_ROLE


def tool_definitions() -> list[dict[str, Any]]:
    integer = {"type": "integer", "minimum": 0}
    positive = {"type": "integer", "minimum": 1}
    string = {"type": "string"}
    fields = {
        "read": (
            {"revision": positive, "offset": integer, "limit": {**positive, "maximum": 32000}},
            [],
        ),
        "write": ({"text": string, "message": string, "reviewed_through": string}, ["text"]),
        "log": ({"offset": integer, "limit": {**positive, "maximum": 100}}, []),
        "diff": (
            {
                "before": positive,
                "after": positive,
                "offset": integer,
                "limit": {**positive, "maximum": 32000},
            },
            [],
        ),
        "restore": ({"revision": positive, "message": string}, ["revision"]),
    }
    descriptions = {
        "read": "Read a retained hypothesis notebook revision on demand.",
        "write": (
            "Replace the complete Markdown notebook and retain a new version with the current "
            "public evidence cursor. No physical action or history edit."
        ),
        "log": "List revision metadata without notebook text.",
        "diff": "Read a paginated textual difference between retained revisions.",
        "restore": "Copy an old revision into a NEW version. Does not undo laboratory operations.",
    }
    return [
        {
            "name": f"notebook_{op}",
            "description": descriptions[op],
            "inputSchema": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": op in {"read", "log", "diff"},
                "destructiveHint": False,
            },
        }
        for op, (props, required) in fields.items()
    ]


class CampaignNotebookServer(ChemWorldMCPServer):
    def __init__(self, public: Path, output: Path) -> None:
        # Intentionally do not initialize the laboratory server's private IPC handles.
        self.public = public
        self.output = output
        self.documents = ExperimentDocumentWorkspace(output, versioned_notebook=True)
        self.documents.initialize()
        self.count = self.documents.verify_authoritative_integrity()["line_count"]

    def sync(self) -> None:
        path = self.public / "operation_history.jsonl"
        if not path.exists():
            if self.count:
                raise ValueError("previously published history is missing")
            return
        data = path.read_bytes()
        # A concurrently appended incomplete last line is not yet a published event.
        lines = data.splitlines() if data.endswith(b"\n") else data.splitlines()[:-1]
        if len(lines) < self.count:
            raise ValueError("published history shrank")
        for line in lines[self.count :]:
            self.documents.append_operation(json.loads(line))
            self.count += 1

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return tool_definitions()

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if request.get("method") == "initialize":
            return self._result(
                request.get("id"),
                {
                    "protocolVersion": request.get("params", {}).get(
                        "protocolVersion", "2025-06-18"
                    ),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "research_notebook", "version": "1.0"},
                    "instructions": (
                        "Versioned scientific notes; optional updates, no physical tools. "
                        + HYPOTHESIS_NOTEBOOK_ROLE
                    ),
                },
            )
        return super()._dispatch(request)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            if name not in {t["name"] for t in tool_definitions()}:
                raise ValueError("unknown notebook tool")
            self.sync()
            payload = self.documents.notebook_tool(name.removeprefix("notebook_"), **arguments)
            result = {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
            }
        except (ValueError, TypeError, KeyError) as exc:
            result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        with (self.output / "notebook-calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": started,
                        "tool": name,
                        "error": result.get("isError", False),
                        "public_event_count": self.count,
                    }
                )
                + "\n"
            )
        return result


class CampaignNotebookMixin:
    """Add the current notebook to a campaign agent without replacing its protocol."""

    def _command(self, *, instructions_path: Path, schema_path: Path) -> list[str]:
        command = super()._command(instructions_path=instructions_path, schema_path=schema_path)
        instructions = (
            instructions_path.read_text(encoding="utf-8") + "\n\n" + HYPOTHESIS_NOTEBOOK_ROLE
        )
        instructions += (
            "\nThe research_notebook tools are authorized in addition to the laboratory and "
            "public numerics. Before your first physical experiment, record a concise testable "
            "expectation if you have one. Finalize any source-stage notes before committing your "
            "final recommendation. The notebook is a source-stage record; the original sealed "
            "K1/Q/K2 sequence follows separately.\n"
        )
        instructions_path.write_text(instructions, encoding="utf-8")
        (self.output / "source-instructions.txt").write_text(instructions, encoding="utf-8")
        config = {
            "command": sys.executable,
            "args": [
                "-m",
                "chemworld.agents.campaign_notebook",
                "--public",
                str(self.workspace.root / "public"),
                "--output",
                str(self.output / "notebook"),
            ],
            "cwd": str(Path(__file__).resolve().parents[3]),
            "required": True,
            "enabled": True,
            "enabled_tools": [t["name"] for t in tool_definitions()],
            "default_tools_approval_mode": "approve",
            "startup_timeout_sec": 60,
            "tool_timeout_sec": 60,
            "supports_parallel_tool_calls": False,
        }
        for key, value in config.items():
            command += ["-c", f"mcp_servers.research_notebook.{key}={json.dumps(value)}"]
        return command


def with_notebook(original: type) -> type:
    return type(f"Notebook{original.__name__}", (CampaignNotebookMixin, original), {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(CampaignNotebookServer(args.public, args.output).run())


if __name__ == "__main__":
    main()
