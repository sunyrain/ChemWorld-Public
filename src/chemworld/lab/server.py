"""Local HTTP server for the provider-free ChemWorld Student Lab."""

from __future__ import annotations

import argparse
import ipaddress
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib.resources import files
from typing import Any
from urllib.parse import urlparse

from chemworld.lab.http_security import LocalServer, RequestRejected, check_local, read_local_json
from chemworld.lab.session import LabSessionManager, task_catalog

MAX_REQUEST_BYTES = 64 * 1024


class LabServer(LocalServer):
    """Loopback-first server that owns in-memory Gym sessions."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, LabHandler)
        self.sessions = LabSessionManager()

    def server_close(self) -> None:
        self.sessions.close_all()
        super().server_close()


class LabHandler(BaseHTTPRequestHandler):
    server: LabServer

    def do_GET(self) -> None:
        try:
            check_local(self)
        except RequestRejected as exc:
            self._error(exc.status, str(exc))
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"status": "ok", "provider_required": False})
            return
        if path == "/api/tasks":
            self._json({"tasks": task_catalog(), "default_task": "reaction-to-assay"})
            return
        if path.startswith("/api/sessions/"):
            try:
                session = self.server.sessions.get(path.split("/")[3])
                if path.endswith("/export"):
                    self._json(session.export())
                else:
                    self._json(session.state())
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self._static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._request_json()
            if path == "/api/sessions":
                task_id = str(body.get("task_id") or "reaction-to-assay")
                seed = int(body["seed"]) if body.get("seed") is not None else None
                session = self.server.sessions.create(task_id, seed)
                self._json(session.state(), status=HTTPStatus.CREATED)
                return
            if path.startswith("/api/sessions/") and path.endswith("/actions"):
                action = body.get("action")
                if not isinstance(action, dict):
                    raise ValueError("request must contain an action object")
                session_id = path.split("/")[3]
                self._json(self.server.sessions.get(session_id).step(action))
                return
            if path.startswith("/api/sessions/") and path.endswith("/replay"):
                self._json(self.server.sessions.get(path.split("/")[3]).replay())
                return
            if path.startswith("/api/sessions/") and path.endswith("/close"):
                self._json(self.server.sessions.close(path.split("/")[3]))
                return
            self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
        except RequestRejected as exc:
            self._error(exc.status, str(exc))
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except (TypeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:  # pragma: no cover - fail closed at the HTTP boundary
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Lab operation failed")

    def _request_json(self) -> dict[str, Any]:
        return read_local_json(self)

    def _static(self, path: str) -> None:
        route = {"": "index.html", "/": "index.html"}.get(path, path.lstrip("/"))
        if not route or "/" in route or route.startswith("."):
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        target = files("chemworld.lab").joinpath("static", route)
        if not target.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get("." + route.rsplit(".", 1)[-1], "application/octet-stream")
        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:",
        )
        self.end_headers()
        self.wfile.write(content)

    def _json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


def serve(host: str = "127.0.0.1", port: int = 8876, *, open_browser: bool = True) -> None:
    """Serve until interrupted; no provider key or network call is required."""

    if host != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError as exc:
            raise ValueError("Student Lab host must be localhost or a loopback address") from exc
        if not loopback:
            raise ValueError("Student Lab only binds to loopback addresses")
    server = LabServer((host, port))
    actual_port = int(server.server_address[1])
    url = f"http://{host}:{actual_port}/"
    print(f"ChemWorld Student Lab: {url}")
    print("Provider-free mode: no API key, model, or external service is used.")
    if open_browser:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
