"""Small loopback HTTP boundary shared by the packaged and checkout labs."""

from __future__ import annotations

import ipaddress
import json
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import socket
from time import monotonic
from typing import Any, NoReturn, cast

MAX_REQUEST_BYTES = 64 * 1024


class RequestRejected(ValueError):  # noqa: N818 -- names a rejected HTTP request
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


class LocalServer(ThreadingHTTPServer):
    """Bound sockets, handler concurrency and write rate; never a remote service."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler]) -> None:
        host = address[0]
        if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
            raise ValueError("Lab only binds to loopback addresses")
        self._slots = threading.BoundedSemaphore(16)
        self._writes: deque[float] = deque()
        self._writes_lock = threading.Lock()
        super().__init__(address, handler)

    def get_request(self) -> tuple[socket, tuple[str, int]]:
        request, address = super().get_request()
        request.settimeout(10)
        return request, address

    def process_request(
        self, request: socket | tuple[bytes, socket], client_address: tuple[str, int]
    ) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                cast(socket, request).sendall(b"HTTP/1.0 503 Busy\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(
        self, request: socket | tuple[bytes, socket], client_address: tuple[str, int]
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def allow_write(self) -> bool:
        now = monotonic()
        with self._writes_lock:
            while self._writes and self._writes[0] <= now - 60:
                self._writes.popleft()
            if len(self._writes) >= 120:
                return False
            self._writes.append(now)
            return True


def check_local(handler: BaseHTTPRequestHandler) -> None:
    hosts = handler.headers.get_all("Host", [])
    origins = handler.headers.get_all("Origin", [])
    port = cast(LocalServer, handler.server).server_port
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if len(hosts) != 1 or hosts[0] not in allowed:
        raise RequestRejected("Local Host required", HTTPStatus.FORBIDDEN)
    if len(origins) > 1 or (origins and origins[0] not in {f"http://{host}" for host in allowed}):
        raise RequestRejected("Local Origin required", HTTPStatus.FORBIDDEN)
    if handler.headers.get("Sec-Fetch-Site") == "cross-site":
        raise RequestRejected("Cross-site request denied", HTTPStatus.FORBIDDEN)


def read_local_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    check_local(handler)
    if handler.headers.get_content_type() != "application/json":
        raise RequestRejected("application/json required", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
    if handler.headers.get_all("Transfer-Encoding"):
        raise RequestRejected("Transfer-Encoding is not supported")
    lengths = handler.headers.get_all("Content-Length", [])
    if len(lengths) != 1:
        raise RequestRejected("One Content-Length is required")
    try:
        length = int(lengths[0])
    except ValueError as exc:
        raise RequestRejected("Invalid Content-Length") from exc
    if length < 0 or length > MAX_REQUEST_BYTES:
        raise RequestRejected("Request too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    if not cast(LocalServer, handler.server).allow_write():
        raise RequestRejected("Write rate exceeded", HTTPStatus.TOO_MANY_REQUESTS)
    raw = handler.rfile.read(length)
    if len(raw) != length:
        raise RequestRejected("Incomplete request body")

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"Non-finite JSON number: {value}")

    body = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    if not isinstance(body, dict):
        raise RequestRejected("Request body must be a JSON object")
    return body
