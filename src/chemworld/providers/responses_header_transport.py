"""Session-owned Responses byte forwarding with standard HTTP request headers."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import threading
import time
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


@contextmanager
def standard_headers_transport(base_url: str, audit: Path, timeout: float):
    target = urlsplit(base_url)
    if target.scheme not in {"http", "https"} or not target.hostname:
        raise ValueError("invalid Responses upstream")
    lock = threading.Lock()
    active = {}
    stopping = threading.Event()
    sequence = 0

    def record(row):
        with lock, audit.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal sequence
            if self.path != "/responses" or stopping.is_set():
                self.send_error(404)
                return
            started = time.monotonic()
            with lock:
                sequence += 1
                request_index = sequence
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                payload = json.loads(body)
                reasoning = payload.get("reasoning")
            except (ValueError, UnicodeError):
                reasoning = None
            record({"request": request_index, "event": "started", "reasoning": reasoning})
            headers = {
                key: self.headers[key]
                for key in ("Authorization", "Content-Type", "Accept", "Content-Encoding")
                if key in self.headers
            }
            headers["User-Agent"] = "ChemWorld/0.2"
            connection_type = (
                http.client.HTTPSConnection
                if target.scheme == "https"
                else http.client.HTTPConnection
            )
            options = {"context": ssl.create_default_context()} if target.scheme == "https" else {}
            connection = connection_type(
                target.hostname, target.port, timeout=min(timeout, 10), **options
            )
            status = None
            failure = None
            response = None
            try:
                connection.connect()
                with lock:
                    active[connection] = connection.sock
                if stopping.is_set():
                    return
                connection.sock.settimeout(timeout)
                path = target.path.rstrip("/") + "/responses"
                connection.request("POST", path, body=body, headers=headers)
                response = connection.getresponse()
                status = response.status
                self.send_response(status)
                self.send_header(
                    "Content-Type", response.getheader("Content-Type", "text/event-stream")
                )
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException) as error:
                failure = type(error).__name__
                if status is None and not stopping.is_set():
                    with suppress(OSError):
                        self.send_error(502, "upstream transport failed")
            finally:
                if response is not None:
                    response.close()
                connection.close()
                with lock:
                    active.pop(connection, None)
                self.close_connection = True
                record(
                    {
                        "request": request_index,
                        "event": "finished",
                        "http_status": status,
                        "failure": failure,
                        "cancelled": stopping.is_set(),
                        "elapsed_s": time.monotonic() - started,
                    }
                )

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.block_on_close = False
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        stopping.set()
        with lock:
            connections = list(active.items())
        for connection, current_socket in connections:
            if current_socket is not None:
                with suppress(OSError):
                    current_socket.shutdown(socket.SHUT_RDWR)
            connection.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
