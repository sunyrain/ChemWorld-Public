"""Publish public ChemWorld views without modifying Gym returns or replay semantics."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any, SupportsFloat
from urllib.parse import urlsplit

import gymnasium as gym

from chemworld.data.logging import to_builtin

SCHEMA_VERSION = "chemworld-blender-public-1"
INSTRUMENT_ASSETS = {"uvvis": "uvvis_01", "ftir": "ftir_01", "ph_meter": "ph_01"}
PUBLIC_COUNTERS = {"cost", "safety_risk", "score"}


def public_frame(env: gym.Env[Any, Any], session_id: str) -> dict[str, Any]:
    """Read only the public view API; unknown measurements remain null."""
    base: Any = env.unwrapped
    view = base.observation_view("tool_json")
    report = base.observation_view("lab_report")
    campaign = view["campaign_state"]
    mask = view["observed_mask"]
    observations = {
        key: value if mask.get(key, False) or key in PUBLIC_COUNTERS else None
        for key, value in view["observation"].items()
    }
    instrument = report.get("instrument_summary", {}).get("instrument")
    operation = report.get("operation_type")
    asset = (
        INSTRUMENT_ASSETS.get(instrument)
        if instrument
        else (
            "reactor_01"
            if operation
            in {
                "add_solvent",
                "add_reagent",
                "add_catalyst",
                "heat",
                "wait",
                "sample",
                "quench",
                "terminate",
            }
            else None
        )
    )
    return to_builtin(
        {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "step": campaign["operation_count"],
            "source": "chemworld_public_api",
            "task_id": view["task"]["task_id"],
            "operation": operation,
            "status": report["status"],
            "observations": observations,
            "observed_mask": mask,
            "campaign": {
                key: campaign[key]
                for key in (
                    "experiment_index",
                    "remaining_budget",
                    "budget",
                    "final_assay_count",
                    "best_score",
                    "done",
                )
            },
            "instrument": instrument,
            "asset_id": asset,
            "mapping_status": "mapped" if asset else "report_only",
            "report_text": report["text"],
            "physical_control_enabled": False,
        }
    )


class BlenderClient:
    """Local presentation and explicit demonstration-logistics client."""

    def __init__(self, url: str = "http://127.0.0.1:8877", timeout: float = 2) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Blender service must be a local HTTP origin")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = None if data is None else json.dumps(data, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            self.url + path, data=payload, headers={"Content-Type": "application/json"}
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def publish(self, frame: dict[str, Any]) -> None:
        self.request("/api/v1/chemworld/frame", frame)

    def release(self, session_id: str) -> None:
        self.request("/api/v1/chemworld/release", {"session_id": session_id})

    def transport(self) -> dict[str, Any]:
        """Move the sealed demonstration carrier, without changing a ChemWorld aliquot."""
        return self.request("/api/v1/environment/demo/transport", {})

    def wait_task(self, task_id: str, timeout: float = 180) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.request("/api/v1/environment/tasks/" + task_id)
            if task["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                return task
            time.sleep(0.25)
        raise TimeoutError("Logistics outcome unknown; query the same task_id before retrying")


class BlenderObserver(gym.Wrapper[Any, Any, Any, Any]):
    """An optional observer, never an alternative experimental executor."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        client: BlenderClient | None = None,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(env)
        self.client = client
        self.sink = sink or (client.publish if client else None)
        self.session_id = uuid.uuid4().hex
        self.projection_error: str | None = None
        self.last_frame: dict[str, Any] | None = None

    def _publish(self) -> None:
        self.last_frame = public_frame(self.env, self.session_id)
        try:
            if self.sink is not None:
                self.sink(self.last_frame)
            self.projection_error = None
        except (OSError, ValueError) as exc:
            # The core operation has already completed: never repeat it on transport failure.
            self.projection_error = str(exc)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if self.last_frame is not None:
            self._release()
            self.session_id = uuid.uuid4().hex
        result = self.env.reset(**kwargs)
        self._publish()
        return result

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        self._publish()
        return result

    def _release(self) -> None:
        if self.client is not None:
            try:
                self.client.release(self.session_id)
            except (OSError, ValueError) as exc:
                self.projection_error = str(exc)

    def close(self) -> None:
        try:
            self._release()
        finally:
            self.env.close()


def attach_blender(env: gym.Env[Any, Any], url: str | None = None) -> gym.Env[Any, Any]:
    """Opt in via an explicit URL or CHEMWORLD_BLENDER_URL; otherwise return env unchanged."""
    target = url if url is not None else os.environ.get("CHEMWORLD_BLENDER_URL")
    return BlenderObserver(env, client=BlenderClient(target)) if target else env
