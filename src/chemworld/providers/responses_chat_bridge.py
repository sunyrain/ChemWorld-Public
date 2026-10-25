"""Local, bounded Responses/SSE transport for a Chat Completions provider.

The upstream stream is collected before output-item events are emitted. Model text,
function arguments, reasoning continuity and reported usage are never fitted or repaired.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class BridgeError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(p.get("type") not in {"input_text", "output_text", "text"} for p in content):
            raise BridgeError("unsupported_content", "Only text content is supported")
        return "\n".join(p["text"] for p in content)
    raise BridgeError("unsupported_content", "Expected text or text content blocks")


def response_usage(usage: dict | None) -> dict | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "input_tokens_details": {
            "cached_tokens": usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", usage.get("prompt_cache_hit_tokens", 0)
            )
        },
        "output_tokens_details": {
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            )
        },
    }


class ResponsesChatBridge:
    def __init__(
        self,
        *,
        api_key: str,
        audit_dir: Path,
        model: str = "zai-org/GLM-5.3",
        upstream_url: str = "https://api.siliconflow.cn/v1/chat/completions",
        max_requests: int = 48,
        max_output_tokens: int = 8192,
        timeout_s: float = 120,
        schema_mode: str = "json_schema",
        port: int = 0,
    ):
        if schema_mode not in {"json_schema", "json_object", "prompt_schema"}:
            raise ValueError("Unknown schema transport mode")
        self.api_key, self.local_token = api_key, secrets.token_hex(32)
        self.audit_dir, self.model, self.upstream_url = audit_dir, model, upstream_url
        audit_dir.mkdir(parents=True, exist_ok=False)
        self.max_requests, self.max_output_tokens = max_requests, max_output_tokens
        self.timeout_s, self.schema_mode = timeout_s, schema_mode
        self.records: list[dict] = []
        self.responses: dict[str, list[dict]] = {}
        self.reasoning: dict[str, str] = {}
        self.items: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.inflight: dict[str, tuple[Any, threading.Event]] = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.cancel_pending()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def cancel_pending(self) -> bool:
        """Cancel upstream work and wait briefly for durable failure receipts."""
        with self.lock:
            pending = list(self.inflight.values())
        deadline = time.monotonic() + 5
        for cancel, _done in pending:
            cancel()
        return all(done.wait(max(0, deadline - time.monotonic())) for _, done in pending)

    def cancel_response(self, response_id: str) -> None:
        with self.lock:
            pending = self.inflight.get(response_id)
        if pending:
            pending[0]()

    def _save(self, name: str, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        for token in (self.api_key, self.local_token):
            encoded = encoded.replace(token, "[REDACTED]")
        (self.audit_dir / name).write_text(encoded + "\n", encoding="utf-8", newline="\n")

    @staticmethod
    def tool_name(name: str, namespace: str | None = None) -> str:
        original = f"{namespace}__{name}" if namespace else name
        return (
            original
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", original)
            else "cw_" + hashlib.sha256(original.encode()).hexdigest()[:20]
        )

    def translate(self, request: dict) -> tuple[dict, dict[str, dict], list[dict]]:
        if request.get("model") != self.model:
            raise BridgeError("model_mismatch", "The bridge is fixed to its configured model")
        supplied = request.get("input", [])
        if isinstance(supplied, str):
            supplied = [{"role": "user", "content": supplied}]
        previous = request.get("previous_response_id")
        if previous and previous not in self.responses:
            raise BridgeError("unknown_previous_response", "Previous response is not retained")
        inputs = deepcopy(self.responses.get(previous, [])) + deepcopy(supplied)
        messages = []
        if request.get("instructions"):
            messages.append({"role": "system", "content": request["instructions"]})
        for original in inputs:
            item = original
            if item.get("type") == "item_reference":
                if item.get("id") not in self.items:
                    raise BridgeError("unknown_item", "Referenced output item is not retained")
                item = self.items[item["id"]]
            kind = item.get("type", "message")
            if kind == "message":
                role = item["role"]
                if role == "developer":
                    role = "system"
                if role not in {"system", "user", "assistant"}:
                    raise BridgeError("unsupported_role", f"Unsupported role: {role}")
                msg = {"role": role, "content": text_content(item.get("content"))}
                if item.get("id") in self.reasoning:
                    msg["reasoning_content"] = self.reasoning[item["id"]]
                messages.append(msg)
            elif kind in {"function_call", "custom_tool_call"}:
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": None})
                msg = messages[-1]
                msg.setdefault("tool_calls", []).append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {
                            "name": self.tool_name(item["name"], item.get("namespace")),
                            "arguments": (
                                item["arguments"]
                                if kind == "function_call"
                                else json.dumps({"input": item["input"]})
                            ),
                        },
                    }
                )
                if item["call_id"] in self.reasoning:
                    msg["reasoning_content"] = self.reasoning[item["call_id"]]
            elif kind in {"function_call_output", "custom_tool_call_output"}:
                output = item["output"]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": text_content(output),
                    }
                )
            else:
                raise BridgeError("unsupported_input", f"Unsupported input item: {kind}")
        names, tools = {}, []

        def register(tool: dict, namespace: str | None = None) -> None:
            if tool.get("type") == "namespace":
                if namespace:
                    raise BridgeError("unsupported_tool", "Nested namespaces are unsupported")
                for child in tool["tools"]:
                    register(child, tool["name"])
                return
            if tool.get("type") not in {"function", "custom"}:
                raise BridgeError("unsupported_tool", "Unsupported declared tool type")
            name = self.tool_name(tool["name"], namespace)
            if name in names:
                raise BridgeError("tool_name_collision", "Flattened tool names must be unique")
            names[name] = {"name": tool["name"], "namespace": namespace, "type": tool["type"]}
            function = {
                "name": name,
                "description": (
                    (f"Tool {namespace}.{tool['name']}. " if namespace else "")
                    + tool.get("description", "")
                ),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            if tool["type"] == "custom":
                function["parameters"] = {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                }
                function["description"] += (
                    "\nPlace the exact freeform tool input in the input string. Format: "
                    + json.dumps(tool.get("format", {"type": "text"}))
                )
            tools.append({"type": "function", "function": function})

        for tool in request.get("tools", []):
            register(tool)
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": min(
                request.get("max_output_tokens") or self.max_output_tokens, self.max_output_tokens
            ),
        }
        for key in ("temperature", "top_p"):
            if key in request:
                body[key] = request[key]
        if tools:
            body["tools"] = tools
            choice = request.get("tool_choice", "auto")
            if isinstance(choice, dict):
                if choice.get("type") not in {"function", "custom"}:
                    raise BridgeError("unsupported_tool_choice", "Expected function tool choice")
                name = self.tool_name(choice["name"], choice.get("namespace"))
                if name not in names:
                    raise BridgeError("unknown_tool_choice", "Tool choice is not declared")
                choice = {"type": "function", "function": {"name": name}}
            body["tool_choice"] = choice
        format_spec = request.get("text", {}).get("format", {})
        if format_spec.get("type") == "json_schema":
            if self.schema_mode == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        k: format_spec[k] for k in ("name", "schema", "strict") if k in format_spec
                    },
                }
            else:
                if self.schema_mode == "json_object":
                    body["response_format"] = {"type": "json_object"}
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": (
                            "Your final answer must be a JSON object satisfying "
                            "this output schema: "
                            + json.dumps(format_spec["schema"], separators=(",", ":"))
                        ),
                    },
                )
        elif format_spec.get("type") == "json_object":
            body["response_format"] = {"type": "json_object"}
        # Some Chat providers accept exactly one leading system message.
        while len(messages) > 1 and messages[0]["role"] == messages[1]["role"] == "system":
            messages[0]["content"] += "\n\n" + messages.pop(1)["content"]
        return body, names, inputs

    def _upstream(self, body: dict, record: dict) -> dict:
        req = urllib.request.Request(
            self.upstream_url,
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        text, reasoning, calls, chunks = [], [], {}, []
        result = {"usage": None, "finish_reason": None}
        expired, cancelled, done = threading.Event(), threading.Event(), threading.Event()
        holder: list[Any] = []
        request_id = str(record.get("response_id", record["request"]))

        def abort(*, timeout: bool = False):
            (expired if timeout else cancelled).set()
            if holder:
                stream = holder[0]
                # shutdown interrupts a concurrent blocking read; close alone may wait on it.
                sock = getattr(getattr(getattr(stream, "fp", None), "raw", None), "_sock", None)
                if sock is not None:
                    with suppress(OSError):
                        sock.shutdown(socket.SHUT_RDWR)

        def check_cancelled():
            if expired.is_set():
                raise BridgeError("upstream_deadline", "Total upstream request deadline exceeded")
            if cancelled.is_set():
                raise BridgeError("upstream_cancelled", "Caller cancelled the upstream request")

        timer = threading.Timer(self.timeout_s, lambda: abort(timeout=True))
        timer.daemon = True
        with self.lock:
            self.inflight[request_id] = (abort, done)
        timer.start()
        journal_path = self.audit_dir / f"{record['request']:03d}-upstream-chunks.jsonl"
        journal = journal_path.open("x", encoding="utf-8", newline="\n")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                holder.append(response)
                check_cancelled()
                record["http_status"] = response.status
                for raw in response:
                    check_cancelled()
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    chunks.append(chunk)
                    encoded = json.dumps(chunk, ensure_ascii=True)
                    for token in (self.api_key, self.local_token):
                        encoded = encoded.replace(token, "[REDACTED]")
                    journal.write(encoded + "\n")
                    journal.flush()
                    record["upstream_chunks"] = len(chunks)
                    if chunk.get("error"):
                        raise BridgeError("upstream_error", json.dumps(chunk["error"]))
                    if chunk.get("model"):
                        result["model"] = chunk["model"]
                        record["model_returned"] = chunk["model"]
                    if chunk.get("usage"):
                        result["usage"] = chunk["usage"]
                        record["usage"] = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            text.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
                        for call in delta.get("tool_calls", []):
                            target = calls.setdefault(
                                call["index"], {"id": "", "name": "", "arguments": ""}
                            )
                            if call.get("id"):
                                target["id"] = call["id"]
                            for key in ("name", "arguments"):
                                target[key] += call.get("function", {}).get(key, "") or ""
                        if choice.get("finish_reason"):
                            result["finish_reason"] = choice["finish_reason"]
                check_cancelled()
        except urllib.error.HTTPError as error:
            record["http_status"] = error.code
            detail = error.read().decode("utf-8", errors="replace")
            self._save(
                f"{record['request']:03d}-upstream-error.json",
                {"status": error.code, "body": detail},
            )
            raise BridgeError(f"upstream_http_{error.code}", detail[:1500]) from error
        except Exception:
            check_cancelled()
            raise
        finally:
            timer.cancel()
            journal.close()
            self._save(f"{record['request']:03d}-upstream-chunks.json", chunks)
            with self.lock:
                self.inflight.pop(request_id, None)
            done.set()
        result.update(
            content="".join(text),
            reasoning_content="".join(reasoning),
            tool_calls=list(calls.values()),
        )
        record.update(
            usage=result["usage"],
            finish_reason=result["finish_reason"],
            model_returned=result.get("model"),
            upstream_chunks=len(chunks),
        )
        if result["finish_reason"] not in {"stop", "tool_calls"}:
            raise BridgeError(
                "upstream_incomplete", "Upstream did not complete: " + str(result["finish_reason"])
            )
        return result

    def complete(self, request: dict, response_id: str) -> dict:
        with self.lock:
            if len(self.records) >= self.max_requests:
                raise BridgeError("request_limit", "Configured request budget exhausted")
            busy = any(r["status"] == "started" for r in self.records)
            record = {
                "request": len(self.records) + 1,
                "status": "started",
                "usage": None,
                "response_id": response_id,
            }
            self.records.append(record)
        start = time.monotonic()
        self._save(f"{record['request']:03d}-responses-request.json", request)
        self._save("ledger.json", self.records)
        try:
            if busy:
                raise BridgeError("request_in_flight", "Previous upstream request is still active")
            body, names, inputs = self.translate(request)
            self._save(f"{record['request']:03d}-chat-request.json", body)
            record.update(
                schema_mode=self.schema_mode,
                requested_reasoning=request.get("reasoning"),
                effective_reasoning="provider_default_not_effort_mapped",
            )
            result = self._upstream(body, record)
            outputs = []
            if result["content"]:
                item_id = "msg_" + secrets.token_hex(12)
                outputs.append(
                    {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "phase": "commentary" if result["tool_calls"] else "final_answer",
                        "content": [
                            {"type": "output_text", "text": result["content"], "annotations": []}
                        ],
                    }
                )
                self.reasoning[item_id] = result["reasoning_content"]
            for call in result["tool_calls"]:
                if call["name"] not in names or not call["id"]:
                    raise BridgeError("invalid_tool_call", "Unknown tool name or missing call id")
                declaration = names[call["name"]]
                item = {
                    "id": "fc_" + secrets.token_hex(12),
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": declaration["name"],
                    "arguments": call["arguments"],
                    "status": "completed",
                }
                if declaration["namespace"]:
                    item["namespace"] = declaration["namespace"]
                if declaration["type"] == "custom":
                    arguments = json.loads(item.pop("arguments"))
                    if not isinstance(arguments, dict) or set(arguments) != {"input"}:
                        raise BridgeError("invalid_custom_input", "Expected only an input field")
                    if not isinstance(arguments["input"], str):
                        raise BridgeError("invalid_custom_input", "Custom input must be text")
                    item.update(type="custom_tool_call", input=arguments["input"])
                outputs.append(item)
                self.reasoning[call["id"]] = result["reasoning_content"]
            for output in outputs:
                self.items[output["id"]] = deepcopy(output)
            if request.get("store", True):
                self.responses[response_id] = inputs + deepcopy(outputs)
            record.update(
                status="completed", output_items=len(outputs), tool_calls=len(result["tool_calls"])
            )
            return {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": self.model,
                "output": outputs,
                "usage": response_usage(result["usage"]),
                "error": None,
                "incomplete_details": None,
                "parallel_tool_calls": False,
            }
        except Exception as error:
            record.update(
                status="failed",
                error_code=getattr(error, "code", type(error).__name__),
                error=str(error).replace(self.api_key, "[REDACTED]"),
            )
            raise
        finally:
            record["wall_seconds"] = round(time.monotonic() - start, 3)
            self._save("ledger.json", self.records)

    def _handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                if not hmac.compare_digest(token, bridge.local_token):
                    self.send_error(401)
                    return
                if self.path.rstrip("/") != "/v1/responses":
                    self.send_error(404)
                    return
                if self.headers.get("Content-Encoding", "identity") != "identity":
                    self.send_error(
                        415, "Request compression is not supported; disable it in Codex"
                    )
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_000_000:
                    self.send_error(413)
                    return
                try:
                    request = json.loads(self.rfile.read(length))
                except ValueError:
                    self.send_error(400)
                    return
                response_id = "resp_" + secrets.token_hex(12)
                pending = {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "in_progress",
                    "model": bridge.model,
                    "output": [],
                }
                streaming = request.get("stream", False)
                sequence, lock, stop = 0, threading.Lock(), threading.Event()

                def event(kind: str, **values):
                    nonlocal sequence
                    with lock:
                        payload = {"type": kind, "sequence_number": sequence, **values}
                        sequence += 1
                        self.wfile.write(
                            ("event: " + kind + "\ndata: " + json.dumps(payload) + "\n\n").encode()
                        )
                        self.wfile.flush()

                def heartbeat():
                    while not stop.wait(15):
                        try:
                            with lock:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                        except OSError:
                            bridge.cancel_response(response_id)
                            return

                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    event("response.created", response=pending)
                    event("response.in_progress", response=pending)
                    threading.Thread(target=heartbeat, daemon=True).start()
                try:
                    response = bridge.complete(request, response_id)
                    stop.set()
                    if streaming:
                        for index, item in enumerate(response["output"]):
                            initial = deepcopy(item)
                            initial["status"] = "in_progress"
                            call_field = {
                                "function_call": "arguments",
                                "custom_tool_call": "input",
                            }.get(item["type"])
                            initial[call_field or "content"] = "" if call_field else []
                            event("response.output_item.added", output_index=index, item=initial)
                            if call_field:
                                event_prefix = (
                                    "response.function_call_arguments"
                                    if call_field == "arguments"
                                    else "response.custom_tool_call_input"
                                )
                                event(
                                    event_prefix + ".delta",
                                    output_index=index,
                                    item_id=item["id"],
                                    delta=item[call_field],
                                )
                                event(
                                    event_prefix + ".done",
                                    output_index=index,
                                    item_id=item["id"],
                                    **{call_field: item[call_field]},
                                )
                            else:
                                part = item["content"][0]
                                event(
                                    "response.content_part.added",
                                    output_index=index,
                                    item_id=item["id"],
                                    content_index=0,
                                    part={"type": "output_text", "text": "", "annotations": []},
                                )
                                event(
                                    "response.output_text.delta",
                                    output_index=index,
                                    item_id=item["id"],
                                    content_index=0,
                                    delta=part["text"],
                                )
                                event(
                                    "response.output_text.done",
                                    output_index=index,
                                    item_id=item["id"],
                                    content_index=0,
                                    text=part["text"],
                                )
                                event(
                                    "response.content_part.done",
                                    output_index=index,
                                    item_id=item["id"],
                                    content_index=0,
                                    part=part,
                                )
                            event("response.output_item.done", output_index=index, item=item)
                        event("response.completed", response=response)
                    else:
                        encoded = json.dumps(response).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(encoded)))
                        self.end_headers()
                        self.wfile.write(encoded)
                except Exception as error:
                    detail = {
                        "code": str(getattr(error, "code", "bridge_error")),
                        "message": str(error).replace(bridge.api_key, "[REDACTED]"),
                    }
                    if streaming:
                        with suppress(OSError):
                            event(
                                "response.failed",
                                response={**pending, "status": "failed", "error": detail},
                            )
                    else:
                        self.send_error(502, detail["code"])
                finally:
                    stop.set()
                    self.close_connection = True

        return Handler
