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
from chemworld.foundation.state import OperationRecord
from chemworld.foundation.state_serialization import restore_state
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID

SESSION_SNAPSHOT_VERSION = "chemworld-session-v2"
_CONTINUATION_FIELDS = (
    "_step_count", "_experiment_index", "_operation_id", "_done", "_last_info",
    "_campaign_id", "_experiment_summaries", "_campaign_resource_current_vessel_started",
    "_last_campaign_resource_receipt", "_current_batch_resource_baseline",
    "_campaign_terminal", "_campaign_terminal_reason", "_right_censored_open_batch",
    "_declared_operation_counts", "_last_observation_noise_provenance", "_lifecycle_history",
)
_RESOURCE_FIELDS = ("_campaign_resource_ledger", "_batch_resource_ledger")
_SNAPSHOT_FIELDS = frozenset({
    "schema_version", "configuration", "execution", "scenario_id", "world_state",
    "observation", "operation_record", "continuation", "rng_state",
    "observation_occurrences", "resource_ledgers",
})


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


def _execution_contract(binding):
    """Canonicalize display notes and neutral optional CPU declarations.

    The reference executor predates optional CPU metadata. Its entirely
    undeclared capability and empty effects equal their legacy absence. Real
    CPU contracts, effects and unknown fields still participate in equality.
    """
    from chemworld.world.domain_modules import NativeExecutionSpec

    cpu_default = NativeExecutionSpec()
    neutral_cpu = all(
        getattr(cpu_default, field) == "not_declared"
        for field in ("lowering", "derivatives", "thread_safety", "snapshot")
    )
    default_declaration = declaration(cpu_default)
    result = dict(binding)
    modules = result.get("modules")
    if isinstance(modules, list):
        canonical = []
        for module in modules:
            if not isinstance(module, Mapping):
                canonical.append(module)
                continue
            values = {key: value for key, value in module.items() if key != "scope_notes"}
            if neutral_cpu and values.get("execution") == default_declaration:
                values.pop("execution")
            if values.get("effects") == []:
                values.pop("effects")
            canonical.append(values)
        result["modules"] = canonical
    return result


def _declared_parameter_binding(session, binding):
    """Omit explicitly opt-in rules only when unused and exactly as installed."""
    from chemworld.world.domain_modules import DeclaredParameterRule

    compiled = session.compiled_composition
    if compiled is None:
        return binding
    used = {(component.kind, name) for component in compiled.spec.components
            for name in component.parameters}
    unused = {(module.module_id, kind, name): declaration(rule)
              for module in session.domain_modules.modules
              for kind, cap in module.components for name, rule in cap.parameters
              if type(rule) is DeclaredParameterRule and (kind, name) not in used}
    result = deepcopy(binding)
    modules = result.get("modules")
    if not isinstance(modules, list):
        return result
    for module in modules:
        if not isinstance(module, Mapping) or not isinstance(module.get("module_id"), str):
            continue
        components = module.get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if (not isinstance(component, list) or len(component) != 2
                    or not isinstance(component[0], str)
                    or not isinstance(component[1], Mapping)):
                continue
            kind, cap = component
            rules = cap.get("parameters")
            if isinstance(rules, list):
                cap["parameters"] = [rule for rule in rules if not (
                    isinstance(rule, list) and len(rule) == 2 and isinstance(rule[0], str)
                    and sum(isinstance(other, list) and len(other) == 2
                            and other[0] == rule[0] for other in rules) == 1
                    and (module.get("module_id"), kind, rule[0]) in unused
                    and rule[1] == unused[(module.get("module_id"), kind, rule[0])])]
    return result


def execution_binding(session):
    return _declared_parameter_binding(session, _execution_contract({
        "runtime": RUNTIME_SEMANTICS_ID,
        "modules": declaration(session.domain_modules.modules),
        "operation_order": list(session.action_codec.operation_types),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "rng": session._rng.bit_generator.__class__.__name__,
    }))


def _execution_binding_matches(session, recorded):
    """Allow added, unused parameter rules in an otherwise identical declaration.

    A composition explicitly names every parameter it consumes. New catalogue
    entries absent from that composition have no default execution effect. Keep
    shared/removed rules, unknown fields and every other execution field strict.
    Without a compiled composition there is no such usage proof.
    """
    from chemworld.world.domain_modules import ParameterRule

    current = execution_binding(session)
    previous = _declared_parameter_binding(session, _execution_contract(recorded))
    if current == previous:
        return True
    compiled = session.compiled_composition
    if compiled is None:
        return False
    used = {(component.kind, name) for component in compiled.spec.components
            for name in component.parameters}
    ordinary_rules = {(kind, name) for kind, cap in session.domain_modules.components().items()
                      for name, rule in cap.parameters if type(rule) is ParameterRule}
    old_modules = previous.get("modules")
    if not isinstance(old_modules, list) or len(old_modules) != len(current["modules"]):
        return False
    for now, old in zip(current["modules"], old_modules, strict=True):
        if not isinstance(old, Mapping) or now.get("module_id") != old.get("module_id"):
            return False
        old_components = old.get("components")
        if (not isinstance(old_components, list)
                or len(old_components) != len(now["components"])):
            return False
        for actual, stored in zip(now["components"], old_components, strict=True):
            if (not isinstance(stored, list) or len(stored) != 2
                    or stored[0] != actual[0] or not isinstance(stored[1], Mapping)):
                return False
            rules = stored[1].get("parameters")
            if not isinstance(rules, list) or any(
                not isinstance(rule, list) or len(rule) != 2 or not isinstance(rule[0], str)
                for rule in rules
            ):
                return False
            names = {rule[0] for rule in rules}
            if len(names) != len(rules):
                return False
            actual[1]["parameters"] = [
                rule for rule in actual[1]["parameters"]
                if rule[0] in names or (actual[0], rule[0]) in used
                or (actual[0], rule[0]) not in ordinary_rules
            ]
    return current == previous


def session_configuration(session):
    """Canonical installed model codecs define configuration equivalence."""
    config = declaration(session._initialization)
    if session.compiled_composition is not None:
        config["composition"] = declaration(session.compiled_composition.spec.to_dict())
    config["seed"] = session.seed
    return config


@dataclass(frozen=True)
class SessionSnapshot:
    payload: dict

    def to_dict(self):
        return deepcopy(self.payload)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, Mapping):
            raise ValueError("session snapshot must be a declared mapping")
        if payload.get("schema_version") != SESSION_SNAPSHOT_VERSION:
            raise ValueError("unsupported session snapshot version; explicit migration required")
        if set(payload) != _SNAPSHOT_FIELDS:
            raise ValueError("session snapshot fields differ from the declared schema")
        for name in ("configuration", "execution", "world_state", "continuation",
                     "rng_state", "resource_ledgers"):
            if not isinstance(payload[name], Mapping):
                raise ValueError(f"session snapshot {name} must be a mapping")
        if set(payload["continuation"]) != set(_CONTINUATION_FIELDS):
            raise ValueError(
                "session snapshot continuation fields differ from the declared contract"
            )
        if set(payload["resource_ledgers"]) != set(_RESOURCE_FIELDS):
            raise ValueError("session snapshot resource fields differ from the declared contract")
        for name in ("_step_count", "_experiment_index", "_operation_id"):
            value = payload["continuation"][name]
            if type(value) is not int or value < 0:
                raise ValueError("session continuation counters require exact nonnegative integers")
        return cls(deepcopy(dict(payload)))

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def migrate_v1(cls, payload):
        """Explicit encoding-only migration when no assembly order was lost.

        Execution bindings still have to match at restoration. Multi-record
        inventories without stored order cannot be reconstructed by guessing.
        """
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != "chemworld-session-v1"
        ):
            raise ValueError("explicit migration requires a v1 session snapshot")
        raw = deepcopy(dict(payload))
        if raw.get("execution", {}).get("runtime") != RUNTIME_SEMANTICS_ID:
            raise ValueError("legacy execution semantics changed; format migration is insufficient")
        raw["schema_version"] = SESSION_SNAPSHOT_VERSION
        raw = cls.from_dict(raw).payload

        def add_order(record, inventory_key, order_key):
            if order_key in record:
                return
            inventory = record.get(inventory_key, {})
            if len(inventory) > 1:
                raise ValueError("legacy snapshot omitted assembly order; source order is required")
            record[order_key] = list(inventory)

        def upgrade_state(state):
            add_order(state, "state_slots", "state_slot_order")
            for sample in state.get("samples", {}).values():
                contents = sample.get("contents")
                if contents is not None:
                    add_order(contents, "phases", "phase_order")
                    add_order(contents, "state_slots", "state_slot_order")
            for local in state.get("inactive_vessels", {}).values():
                upgrade_state(local)

        upgrade_state(raw["world_state"])
        composition = raw["configuration"].get("composition")
        if isinstance(composition, Mapping) and composition.get("physics") is not None:
            for state in composition["physics"]["initial_states"].values():
                upgrade_state(state)
        return cls.from_dict(raw)


def capture_session(session):
    config = session_configuration(session)
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
            for name in _RESOURCE_FIELDS
        },
    })


def restore_continuation(session, snapshot):
    payload = SessionSnapshot.from_dict(snapshot.to_dict()).payload
    if not _execution_binding_matches(session, payload["execution"]):
        raise ValueError("session execution contracts changed; explicit migration required")
    state = restore_state(payload["world_state"], require_order=True)
    # State readers migrate historical equipment/material representations at
    # their existing boundary; no legacy truth copy is retained in the session.
    report = session.constitution.check_state(state)
    if not report.passed:
        raise ValueError("session snapshot contains an invalid physical state")
    session.runtime.domain_services.reaction_thermal.advancer.validate_snapshot_state(state)
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
