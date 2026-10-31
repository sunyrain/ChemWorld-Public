"""Public calculator isolation and accounting, independent of study launchers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chemworld.agents.diagnostic_numerics import calculate


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "open('api.md')",
        "(1).__class__",
        "[1][0]",
        "x+1",
        "[x for x in [1]]",
        "True",
        "1/0",
        "exp(10000)",
        "linspace(0,1,4097)",
        "linspace(0,1,4096)*array([[1],[2]])",
        "clip(linspace(0,1,4096),[[0],[1]],1)",
        "linspace([0,1],1,4096)",
    ],
)
def test_numerics_rejects_private_and_unbounded_paths(expression: str) -> None:
    with pytest.raises((ValueError, TypeError, ArithmeticError)):
        calculate(expression)


def test_public_numerical_reference() -> None:
    # A generic known public table, independent of hidden B3 simulator parameters.
    assert calculate("lstsq([[1,0],[1,1],[1,2]],[2,5,8])") == pytest.approx([2, 3])
    assert calculate("mean(abs(array([1,2,3])-array([2,2,5])))") == pytest.approx(1)
    assert calculate("log(exp(0.75))") == pytest.approx(0.75)


def test_stdio_mcp_and_reconnect_attempt_accounting(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    command = [
        sys.executable,
        "-m",
        "chemworld.agents.diagnostic_numerics",
        "--audit",
        str(audit),
        "--limit",
        "1",
    ]
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "calculate", "arguments": {"expression": "1+2"}},
        },
    ]
    result = subprocess.run(
        command,
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["result"]["serverInfo"]["name"] == "public_numerics"
    assert [t["name"] for t in rows[1]["result"]["tools"]] == ["calculate"]
    assert json.loads(rows[2]["result"]["content"][0]["text"])["value"] == 3
    again = subprocess.run(
        command,
        input=json.dumps(requests[2]) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    assert json.loads(again.stdout)["result"]["isError"]
    assert len(audit.read_text().splitlines()) == 2
