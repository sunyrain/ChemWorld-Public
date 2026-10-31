import io
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chemworld.providers.responses_chat_bridge import (
    BridgeError,
    ResponsesChatBridge,
    response_usage,
)


def test_tool_return_preserves_reasoning_arguments_and_roles(tmp_path):
    with ResponsesChatBridge(api_key="secret", audit_dir=tmp_path / "audit") as bridge:
        bridge.reasoning["call1"] = "private provider reasoning"
        request = {
            "model": bridge.model,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "public contract"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call1",
                    "name": "calculate",
                    "arguments": '{"x":19}',
                },
                {"type": "function_call_output", "call_id": "call1", "output": "42"},
            ],
            "tools": [{"type": "function", "name": "calculate", "parameters": {"type": "object"}}],
        }
        body, _, _ = bridge.translate(request)
        assert body["messages"][0] == {"role": "system", "content": "public contract"}
        assert body["messages"][1]["reasoning_content"] == "private provider reasoning"
        assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"x":19}'
        assert body["messages"][2] == {"role": "tool", "tool_call_id": "call1", "content": "42"}
        assert response_usage(None) is None


def test_sse_roundtrip_previous_id_auth_and_usage(monkeypatch, tmp_path):
    seen = []

    def upstream(self, body, record):
        seen.append(body)
        return {
            "content": "answer",
            "reasoning_content": "private",
            "tool_calls": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }

    monkeypatch.setattr(ResponsesChatBridge, "_upstream", upstream)
    with ResponsesChatBridge(api_key="upstream-secret", audit_dir=tmp_path / "audit") as bridge:

        def post(data, token):
            req = urllib.request.Request(
                bridge.base_url + "/responses",
                data=json.dumps(data).encode(),
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as response:
                return response.read().decode()

        with pytest.raises(urllib.error.HTTPError) as error:
            post({"model": bridge.model}, "incorrect-token")
        assert error.value.code == 401
        wire = post({"model": bridge.model, "input": "first", "stream": True}, bridge.local_token)
        events = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")]
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
        assert events[-1]["type"] == "response.completed"
        completed = events[-1]["response"]
        assert completed["usage"]["output_tokens_details"]["reasoning_tokens"] == 3
        assert "private" not in wire
        post(
            {"model": bridge.model, "previous_response_id": completed["id"], "input": "second"},
            bridge.local_token,
        )
        assert [m["role"] for m in seen[-1]["messages"]] == ["user", "assistant", "user"]
        assert seen[-1]["messages"][1]["reasoning_content"] == "private"


def test_upstream_failure_stays_failed_no_retry_or_usage_fabrication(monkeypatch, tmp_path):
    calls = []

    def failure(self, body, record):
        calls.append(body)
        raise BridgeError("upstream_http_429", "rate limited")

    monkeypatch.setattr(ResponsesChatBridge, "_upstream", failure)
    with ResponsesChatBridge(api_key="secret", audit_dir=tmp_path / "audit") as bridge:
        with pytest.raises(BridgeError, match="rate limited"):
            bridge.complete({"model": bridge.model, "input": "test"}, "r1")
        assert len(calls) == 1
        assert bridge.records[0]["status"] == "failed"
        assert bridge.records[0]["usage"] is None
        assert "r1" not in bridge.responses


@pytest.mark.parametrize("kind", ["function", "custom"])
def test_namespaced_tool_roundtrip_keeps_identity_input_and_reasoning(kind, monkeypatch, tmp_path):
    original = "*** Begin Patch\n*** End Patch\n" if kind == "custom" else '{"expression":"19+23"}'
    argument = json.dumps({"input": original}) if kind == "custom" else original

    def upstream(self, body, record):
        return {
            "content": "",
            "reasoning_content": "reasoning continuity",
            "usage": None,
            "tool_calls": [{"id": "c1", "name": "numerics__calculate", "arguments": argument}],
        }

    monkeypatch.setattr(ResponsesChatBridge, "_upstream", upstream)
    with ResponsesChatBridge(api_key="secret", audit_dir=tmp_path / "audit") as bridge:
        request = {
            "model": bridge.model,
            "input": "compute",
            "tools": [
                {
                    "type": "namespace",
                    "name": "numerics",
                    "tools": [
                        {"type": kind, "name": "calculate", "parameters": {"type": "object"}}
                    ],
                }
            ],
        }
        output = bridge.complete(request, "r1")["output"][0]
        assert output["namespace"] == "numerics"
        assert output["name"] == "calculate"
        assert output["call_id"] == "c1"
        assert output["input" if kind == "custom" else "arguments"] == original
        request.update(
            previous_response_id="r1",
            input=[
                {
                    "type": "custom_tool_call_output"
                    if kind == "custom"
                    else "function_call_output",
                    "call_id": "c1",
                    "output": "42",
                }
            ],
        )
        body, _, _ = bridge.translate(request)
        call = body["messages"][1]["tool_calls"][0]
        assert call["function"] == {"name": "numerics__calculate", "arguments": argument}
        assert body["messages"][1]["reasoning_content"] == "reasoning continuity"
        assert body["messages"][2] == {"role": "tool", "tool_call_id": "c1", "content": "42"}






@pytest.mark.parametrize("error_after_usage", [False, True])
def test_upstream_stream_retains_fragmented_calls_and_known_usage(
    monkeypatch, tmp_path, error_after_usage
):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    chunks = [
        {
            "model": "zai-org/GLM-5.3",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "calculate", "arguments": '{"x":'},
                            }
                        ]
                    }
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "42}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"usage": usage, "choices": []},
    ]
    if error_after_usage:
        chunks.append({"error": {"message": "interrupted after usage"}})
    response = io.BytesIO(
        ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
    )
    response.status = 200
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_kw: response)
    with ResponsesChatBridge(api_key="secret", audit_dir=tmp_path / "audit") as bridge:
        record = {"request": 1}
        if error_after_usage:
            with pytest.raises(BridgeError, match="interrupted"):
                bridge._upstream({}, record)
        else:
            result = bridge._upstream({}, record)
            assert result["tool_calls"] == [
                {"id": "c1", "name": "calculate", "arguments": '{"x":42}'}
            ]
        assert record["usage"] == usage


@pytest.mark.parametrize("mode", ["deadline", "cancel"])
def test_keepalives_cannot_extend_deadline_and_pending_requests_are_cancelled(tmp_path, mode):
    stopped = threading.Event()
    request_count = []
    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request_count.append(1)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(
                    ("data: " + json.dumps({"usage": usage, "choices": []}) + "\n\n").encode()
                )
                self.wfile.flush()
                while not stopped.wait(0.01):
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except OSError:
                return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with (
            ResponsesChatBridge(
                api_key="secret",
                audit_dir=tmp_path / "audit",
                upstream_url=f"http://127.0.0.1:{server.server_port}/chat/completions",
                timeout_s=0.3 if mode == "deadline" else 3,
            ) as bridge,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            started = time.monotonic()
            request = {"model": bridge.model, "input": "test"}
            future = pool.submit(bridge.complete, request, "r1")
            if mode == "cancel":
                while not bridge.records or bridge.records[0].get("usage") is None:
                    assert time.monotonic() - started < 2
                    stopped.wait(0.01)
                with pytest.raises(BridgeError, match="still active"):
                    bridge.complete(request, "r2")
                assert bridge.cancel_pending()
            with pytest.raises(BridgeError) as error:
                future.result(timeout=2)
            assert error.value.code == (
                "upstream_deadline" if mode == "deadline" else "upstream_cancelled"
            )
            assert time.monotonic() - started < 2
            assert len(request_count) == 1
            assert bridge.records[0]["usage"] == usage
            assert bridge.records[0]["status"] == "failed"
            journal = (tmp_path / "audit/001-upstream-chunks.jsonl").read_text()
            assert json.loads(journal.splitlines()[0])["usage"] == usage
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_prompt_schema_keeps_tools_and_original_schema_without_upstream_json_constraint(tmp_path):
    schema = {"type": "object", "properties": {"sum": {"type": "number"}}, "required": ["sum"]}
    with ResponsesChatBridge(
        api_key="secret", audit_dir=tmp_path / "audit", schema_mode="prompt_schema"
    ) as bridge:
        request = {
            "model": bridge.model,
            "input": "Use calculate and then submit the answer.",
            "text": {"format": {"type": "json_schema", "name": "final", "schema": schema}},
            "tools": [{"type": "function", "name": "calculate", "parameters": schema}],
        }
        body, _, _ = bridge.translate(request)
        assert "response_format" not in body
        assert body["tool_choice"] == "auto"
        assert body["tools"][0]["function"]["parameters"] == schema
        content = body["messages"][0]["content"]
        assert json.loads(content[content.index("{") :]) == schema
        assert request["input"] == "Use calculate and then submit the answer."


def test_prompt_schema_merges_only_leading_system_messages_without_losing_the_contract(tmp_path):
    with ResponsesChatBridge(
        api_key="secret", audit_dir=tmp_path / "audit", schema_mode="prompt_schema"
    ) as bridge:
        request = {
            "model": bridge.model,
            "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
            "input": [
                {"role": "developer", "content": "Preserve this tool contract."},
                {"role": "user", "content": "Question"},
            ],
        }
        body, _, _ = bridge.translate(request)
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["messages"][0]["content"].endswith("\n\nPreserve this tool contract.")
        assert '{"type":"object"}' in body["messages"][0]["content"]
        assert body["messages"][1]["content"] == "Question"
