"""Private, versioned continuation data; never an agent-visible observation."""

import json
import platform
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np
import scipy

from chemworld.campaign_resources import CampaignResourceLedger
from chemworld.foundation.balances import MaterialExchange, TransitionBalance
from chemworld.foundation.observations import Observation
from chemworld.foundation.state import OperationRecord, WorldState
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID

SESSION_SNAPSHOT_VERSION = "chemworld-session-v1"
_CONTINUATION_FIELDS = (
    "_step_count", "_experiment_index", "_operation_id", "_done", "_last_info",
    "_campaign_id", "_experiment_summaries", "_campaign_resource_current_vessel_started",
    "_last_campaign_resource_receipt", "_current_batch_resource_baseline",
    "_campaign_terminal", "_campaign_terminal_reason", "_right_censored_open_batch",
    "_declared_operation_counts", "_last_observation_noise_provenance", "_lifecycle_history",
)


def declaration(value):
    """Record installed contracts and callable identities, without importing code on read."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: declaration(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): declaration(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [declaration(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(declaration(item) for item in value)
    if callable(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, np.generic):
        return value.item()
    return value


def execution_binding(session):
    return {
        "runtime": RUNTIME_SEMANTICS_ID,
        "modules": declaration(session.domain_modules.modules),
        "operation_order": list(session.action_codec.operation_types),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "rng": session._rng.bit_generator.__class__.__name__,
    }


@dataclass(frozen=True)
class SessionSnapshot:
    payload: dict

    def to_dict(self):
        return deepcopy(self.payload)

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != SESSION_SNAPSHOT_VERSION:
            raise ValueError("unsupported session snapshot version; explicit migration required")
        return cls(deepcopy(dict(payload)))

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.payload, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def capture_session(session):
    config = declaration(session._initialization)
    config["seed"] = session.seed
    record = session._last_operation_record
    return SessionSnapshot({
        "schema_version": SESSION_SNAPSHOT_VERSION,
        "configuration": config,
        "execution": execution_binding(session),
        "scenario_id": session.scenario_spec.scenario_id,
        "world_state": session._state.to_dict(include_hidden=True),
        "observation": session.observe().native_record(),
        "operation_record": None if record is None else asdict(record),
        "continuation": {name: deepcopy(getattr(session, name)) for name in _CONTINUATION_FIELDS},
        "rng_state": deepcopy(session._rng.bit_generator.state),
        "observation_occurrences": [
            [list(key), count] for key, count in sorted(session._observation_occurrences.items())
        ],
        "resource_ledgers": {
            name: None if getattr(session, name) is None else getattr(session, name).snapshot()
            for name in ("_campaign_resource_ledger", "_batch_resource_ledger")
        },
    })


def restore_continuation(session, snapshot):
    payload = SessionSnapshot.from_dict(snapshot.to_dict()).payload
    if execution_binding(session) != payload["execution"]:
        raise ValueError("session execution contracts changed; explicit migration required")
    state = WorldState.from_dict(payload["world_state"])
    # State readers migrate historical equipment/material representations at
    # their existing boundary; no legacy truth copy is retained in the session.
    report = session.constitution.check_state(state)
    if not report.passed:
        raise ValueError("session snapshot contains an invalid physical state")
    values = deepcopy(payload["continuation"])
    for name, value in payload["resource_ledgers"].items():
        values[name] = None if value is None else CampaignResourceLedger.from_snapshot(value)
    raw = payload["operation_record"]
    if raw is None:
        record = None
    else:
        raw = deepcopy(raw)
        balance = raw.pop("material_balance")
        balance["exchanges"] = tuple(
            MaterialExchange(**{**item, "amounts": tuple(tuple(p) for p in item["amounts"])})
            for item in balance["exchanges"]
        )
        balance["stock_consumption"] = tuple(tuple(p) for p in balance["stock_consumption"])
        balance["unmodeled_quantities"] = tuple(balance["unmodeled_quantities"])
        record = OperationRecord(**raw, material_balance=TransitionBalance(**balance))
    values.update(
        _state=state,
        _last_reading=Observation.from_native_record(payload["observation"]),
        _last_operation_record=record,
        _observation_occurrences={tuple(key): n for key, n in payload["observation_occurrences"]},
    )
    session._rng.bit_generator.state = deepcopy(payload["rng_state"])
    session.__dict__.update(values)
