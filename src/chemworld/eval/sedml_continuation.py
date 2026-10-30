"""Private JSON continuation for a complete finite experiment, at course boundaries.

The task stack contains control positions and historical observations. All evolving
physical state, clocks, pending events and RNG remain in the existing native sessions.
"""

import hashlib
import json
from dataclasses import asdict, replace
from math import isfinite
from uuid import UUID, uuid4, uuid5

from chemworld.agents.base import HistoryRecord
from chemworld.data.sedml_declarations import boolean, records
from chemworld.data.sedml_reports import SEDMLArray
from chemworld.data.xml_inputs import bounded_xml
from chemworld.eval.sedml_experiment import (
    SEDMLCourseResult,
    SEDMLExecutionError,
    SEDMLExperimentResult,
    _run_course,
    _TaskRunner,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sedml_protocol import SEDMLProtocol

EXPERIMENT_SNAPSHOT_VERSION = "chemworld-sedml-experiment-v1"
_FIELDS = {
    "format",
    "protocol",
    "seed",
    "noise_mode",
    "identity_namespace",
    "root_index",
    "courses_completed",
    "stack",
    "runner",
    "arrays",
    "history",
    "iterations",
    "snapshots",
    "status",
    "error",
}


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _encode(arrays):
    return {
        model: {key: array.to_dict() for key, array in channels.items()}
        for model, channels in arrays.items()
    }


def _array(value):
    if not isinstance(value, dict) or set(value) != {
        "values",
        "shape",
        "axes",
        "sample_keys",
        "unit",
    }:
        raise ValueError("unknown experiment checkpoint array fields")
    return SEDMLArray(
        tuple(value["values"]),
        tuple(value["shape"]),
        tuple(value["axes"]),
        tuple(tuple(key) for key in value["sample_keys"]),
        value["unit"],
    )


def _decode(arrays):
    return {
        model: {key: _array(value) for key, value in channels.items()}
        for model, channels in arrays.items()
    }


def _world_matches(protocol, model_id, world):
    original = protocol.worlds[model_id]
    if original.to_dict() == world.to_dict():
        return True
    spec = original.physics.models_of("sbml")["formal_workspace"]
    restored = world.physics.models_of("sbml").get("formal_workspace")
    if restored is None:
        return False
    namespace = "{http://www.sbml.org/sbml/level3/version2/core}"
    tree = bounded_xml(restored.model.to_xml())
    baseline = {
        node.get("id"): node.get("value")
        for node in bounded_xml(spec.model.to_xml()).findall(
            namespace + "model/" + namespace + "listOfParameters/" + namespace + "parameter"
        )
    }
    inputs = {
        node.get("id"): float(node.get("value"))
        for node in tree.findall(
            namespace + "model/" + namespace + "listOfParameters/" + namespace + "parameter"
        )
        if node.get("id") in spec.controlled_parameters
        and node.get("value") != baseline[node.get("id")]
    }
    # Only the protocol's declared reset inputs may vary its immutable world recipe.
    return protocol.world_with_parameters(model_id, inputs).to_dict() == world.to_dict()


class SEDMLExperimentSession:
    """Advance whole tasks without repeating completed courses after private JSON restore.

    ``advance(max_courses=1)`` pauses after a simple time course and its output
    bookkeeping. Model control/acquisition remains accounted native execution.
    ``to_dict`` is private continuation data, never an agent-visible observation.
    """

    def __init__(self, protocol, *, seed=0, observation_noise_mode="sequential"):
        if (
            not isinstance(protocol, SEDMLProtocol)
            or type(seed) is not int
            or seed < 0
            or observation_noise_mode not in ("sequential", "keyed")
        ):
            raise ValueError("experiment requires a compiled protocol, seed and native noise mode")
        self.protocol, self.seed, self.noise_mode = protocol, seed, observation_noise_mode
        self.identity_namespace = uuid4().hex
        selected = {identity for identity, _, _ in protocol.reports.bindings.values()}
        self.roots = tuple(
            identity
            for identity in protocol.tasks.nodes
            if identity in (selected or set(protocol.tasks.roots))
        )
        self.root_index = self.courses_completed = 0
        self.stack, self.history, self.iterations = [], [], []
        self.arrays, self.snapshots = {}, {}
        self.runner = None
        self.status, self.error = "ready", None

    @property
    def complete(self):
        return self.status == "complete"

    def progress(self):
        return {
            "status": self.status,
            "completed_courses": self.courses_completed,
            "total_courses": sum(self.protocol.tasks.invocations[key] for key in self.roots),
            "completed_roots": self.root_index,
            "total_roots": len(self.roots),
        }

    def _push(self, identity):
        self.stack.append(
            {
                "task": identity,
                "phase": "start",
                "index": 0,
                "sub_index": 0,
                "ranges": {},
                "parts": [],
            }
        )

    def _finish_root(self, channels):
        identity = self.roots[self.root_index]
        runner = self.runner
        for key, (task_id, model_id, quantity) in self.protocol.reports.bindings.items():
            if task_id == identity:
                spec = self.protocol.worlds[model_id].physics.models_of("sbml")["formal_workspace"]
                course = SEDMLCourseResult(
                    runner.runtimes[model_id], channels[model_id], (), spec.initial_model_time
                )
                self.arrays[key] = course.source_array(quantity)
        offset = len(self.history)
        self.history.extend(
            replace(record, step=offset + i + 1) for i, record in enumerate(runner.history)
        )
        self.iterations.extend(runner.iterations)
        self.snapshots[identity] = {
            model: runtime.snapshot().to_dict() for model, runtime in runner.runtimes.items()
        }
        self.runner = None
        self.root_index += 1
        if self.root_index == len(self.roots):
            self.protocol.reports.evaluate(self.arrays)  # domain errors are terminal and retained
            self.status = "complete"

    def _accept(self, channels):
        self.stack.pop()
        if not self.stack:
            self._finish_root(channels)
            return
        frame = self.stack[-1]
        prefix = (
            "repeat:" + frame["task"],
            frame["index"],
            "subtask",
            frame["sub_index"],
            *(value for pair in sorted(frame["ranges"].items()) for value in pair),
        )
        frame["parts"].append(
            {
                model: {
                    key: SEDMLArray(
                        array.values,
                        array.shape,
                        array.axes,
                        tuple((*prefix, *sample) for sample in array.sample_keys),
                        array.unit,
                    )
                    for key, array in arrays.items()
                }
                for model, arrays in channels.items()
            }
        )
        frame["sub_index"] += 1
        frame["phase"] = "subtask"

    def _collect(self, frame):
        result = {}
        for model_id, (shape, axes) in self.protocol.layouts[frame["task"]].items():
            parts = [part[model_id] for part in frame["parts"] if model_id in part]
            result[model_id] = {
                key: SEDMLArray(
                    tuple(value for part in parts for value in part[key].values),
                    shape,
                    axes,
                    tuple(sample for part in parts for sample in part[key].sample_keys),
                    parts[0][key].unit,
                )
                for key in parts[0]
            }
        self._accept(result)

    def _step(self):
        if self.runner is None:
            identity = self.roots[self.root_index]
            self.runner = _TaskRunner(
                self.protocol, identity, self.seed, self.noise_mode, self.identity_namespace
            )
            self._push(identity)
        frame = self.stack[-1]
        identity = frame["task"]
        node = self.protocol.tasks.nodes[identity]
        if node.local_name == "task":
            model_id = node.attributes["modelReference"]
            runtime = self.runner.session(model_id)
            simulation = self.protocol.tasks.simulations[node.attributes["simulationReference"]]
            try:
                result = _run_course(self.runner.worlds[model_id], simulation, runtime=runtime)
            except SEDMLExecutionError as exc:
                self.runner._history(exc.history, model_id, identity)
                raise
            self.runner._history(result.history, model_id, identity)
            self.courses_completed += 1
            self._accept({model_id: result.native_channels})
            return True
        if frame["phase"] == "start":
            frame["phase"] = "iteration"
        if frame["phase"] == "iteration":
            if frame["index"] == self.protocol.tasks.ranges[identity].count:
                self._collect(frame)
                return False
            reset = boolean(node.attributes["resetModel"])
            if reset:
                for model in sorted(self.protocol.tasks.model_ids[identity]):
                    self.runner.reset(model)
            self.runner._readings.clear()
            ranges = self.protocol.tasks.ranges[identity].values_at(
                frame["index"], model_value=self.runner.model_value
            )
            frame["ranges"] = dict(ranges)
            self.runner.iterations.append(
                {"task_reference": identity, "index": frame["index"], "range_values": dict(ranges)}
            )
            self.runner.changes(records(node, "listOfChanges"), ranges, identity, initialize=reset)
            frame["sub_index"], frame["phase"] = 0, "subtask"
        if frame["phase"] == "subtask":
            children = self.protocol.tasks.sub_tasks[identity]
            if frame["sub_index"] == len(children):
                frame["index"] += 1
                frame["phase"] = "iteration"
            else:
                child = children[frame["sub_index"]]
                self.runner.changes(records(child, "listOfChanges"), frame["ranges"], identity)
                frame["phase"] = "collect"
                self._push(child.attributes["task"])
        return False

    def _saved_sessions(self):
        return (
            {}
            if self.runner is None
            else {
                model: runtime.snapshot().to_dict()
                for model, runtime in self.runner.runtimes.items()
            }
        )

    def advance(self, *, max_courses=None):
        if max_courses is not None and (type(max_courses) is not int or max_courses < 1):
            raise ValueError("course limit must be a positive integer")
        if self.status == "failed":
            raise RuntimeError("failed experiment continuation is terminal")
        started = self.courses_completed
        try:
            while not self.complete and (
                max_courses is None or self.courses_completed - started < max_courses
            ):
                if self._step():
                    # Finish output-only frames without starting another iteration or action.
                    while self.stack:
                        frame = self.stack[-1]
                        task = frame["task"]
                        if (
                            frame["phase"] != "subtask"
                            or frame["sub_index"] != len(self.protocol.tasks.sub_tasks[task])
                            or frame["index"] + 1 != self.protocol.tasks.ranges[task].count
                        ):
                            break
                        self._collect(frame)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            self.status, self.error = "failed", str(exc)
            active = () if self.runner is None else self.runner.history
            history = (
                *self.history,
                *(
                    replace(record, step=len(self.history) + i + 1)
                    for i, record in enumerate(active)
                ),
            )
            runtime = (
                None if self.runner is None else next(reversed(self.runner.runtimes.values()), None)
            )
            error = SEDMLExecutionError(str(exc), history=history, runtime=runtime)
            error.snapshots = dict(self.snapshots)
            if self.runner is not None:
                error.snapshots[self.runner.root_id] = self._saved_sessions()
            if error.snapshot is None and self.snapshots:
                final_models = next(reversed(self.snapshots.values()))
                error.snapshot = next(reversed(final_models.values()), None)
            error.iterations = (*self.iterations, *(self.runner.iterations if self.runner else ()))
            error.variable_arrays = {
                str(key): value.to_dict() for key, value in self.arrays.items()
            }
            error.experiment_checkpoint = self.to_dict()
            raise error from exc
        return self.progress()

    def result(self):
        if not self.complete:
            raise ValueError("experiment reports are final only after all requested tasks complete")
        return SEDMLExperimentResult(
            self.protocol.reports.evaluate(self.arrays),
            tuple(self.history),
            tuple(self.iterations),
            _plain(self.snapshots),
        )

    def to_dict(self):
        runner = (
            None
            if self.runner is None
            else {
                "root": self.runner.root_id,
                "sessions": self._saved_sessions(),
                "epochs": self.runner.epochs,
                "readings": self.runner._readings,
                "history": [asdict(record) for record in self.runner.history],
                "iterations": self.runner.iterations,
            }
        )
        return _plain(
            {
                "format": EXPERIMENT_SNAPSHOT_VERSION,
                "protocol": self.protocol.to_dict(),
                "seed": self.seed,
                "noise_mode": self.noise_mode,
                "identity_namespace": self.identity_namespace,
                "root_index": self.root_index,
                "courses_completed": self.courses_completed,
                "stack": [
                    {**frame, "parts": [_encode(part) for part in frame["parts"]]}
                    for frame in self.stack
                ],
                "runner": runner,
                "arrays": [
                    {"generator": key[0], "variable": key[1], "data": value.to_dict()}
                    for key, value in self.arrays.items()
                ],
                "history": [asdict(record) for record in self.history],
                "iterations": self.iterations,
                "snapshots": self.snapshots,
                "status": self.status,
                "error": self.error,
            }
        )

    @classmethod
    def from_dict(cls, payload):
        try:
            return cls._from_dict(payload)
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError("malformed experiment continuation structure") from exc

    @classmethod
    def _from_dict(cls, payload):
        if (
            not isinstance(payload, dict)
            or set(payload) != _FIELDS
            or payload["format"] != EXPERIMENT_SNAPSHOT_VERSION
        ):
            raise ValueError("unknown experiment continuation fields or version")
        payload = _plain(payload)
        if (
            any(
                not isinstance(payload[name], list)
                for name in ("stack", "arrays", "history", "iterations")
            )
            or not isinstance(payload["snapshots"], dict)
            or (payload["runner"] is not None and not isinstance(payload["runner"], dict))
            or (payload["error"] is not None and not isinstance(payload["error"], str))
        ):
            raise ValueError("checkpoint collections and error must use their declared JSON types")
        self = cls(
            SEDMLProtocol.from_dict(payload["protocol"]),
            seed=payload["seed"],
            observation_noise_mode=payload["noise_mode"],
        )
        namespace = payload["identity_namespace"]
        if not isinstance(namespace, str) or UUID(hex=namespace).hex != namespace:
            raise ValueError("private experiment identity namespace must be a UUID")
        self.identity_namespace = namespace
        for name in ("root_index", "courses_completed"):
            if type(payload[name]) is not int or payload[name] < 0:
                raise ValueError("experiment continuation cursor must contain nonnegative integers")
            setattr(self, name, payload[name])
        if self.root_index > len(self.roots) or payload["status"] not in {
            "ready",
            "complete",
            "failed",
        }:
            raise ValueError("experiment continuation status/root cursor is invalid")
        self.status, self.error = payload["status"], payload["error"]
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("checkpoint error does not match its terminal status")
        if self.complete != (self.root_index == len(self.roots)) and self.status != "failed":
            raise ValueError("experiment continuation completion does not match its root cursor")
        self.history = [HistoryRecord(**record) for record in payload["history"]]
        self.iterations, self.snapshots = payload["iterations"], payload["snapshots"]
        for record in payload["arrays"]:
            if set(record) != {"generator", "variable", "data"}:
                raise ValueError("unknown checkpoint report binding fields")
            key = record["generator"], record["variable"]
            if key in self.arrays or key not in self.protocol.reports.bindings:
                raise ValueError("unknown or duplicate checkpoint report binding")
            task, model, quantity = self.protocol.reports.bindings[key]
            array = _array(record["data"])
            if (
                task not in self.roots[: self.root_index]
                or array.unit != quantity.source_unit
                or (array.shape, array.axes) != self.protocol.layouts[task][model]
            ):
                raise ValueError("checkpoint report binding differs from the compiled source")
            self.arrays[key] = array
        expected_keys = {
            key
            for key, (task, _, _) in self.protocol.reports.bindings.items()
            if task in self.roots[: self.root_index]
        }
        if set(self.arrays) != expected_keys or set(self.snapshots) != set(
            self.roots[: self.root_index]
        ):
            raise ValueError("checkpoint must retain every completed root and report input")
        self.stack = [
            {**frame, "parts": [_decode(part) for part in frame["parts"]]}
            for frame in payload["stack"]
        ]
        self._restore_runner(payload["runner"])
        self._validate_stack()
        for root, models in self.snapshots.items():
            if not isinstance(models, dict) or set(models) != set(
                self.protocol.tasks.model_ids[root]
            ):
                raise ValueError("completed root snapshots do not match its model identities")
            for model, snapshot in models.items():
                restored = ChemWorldRuntime.from_snapshot(snapshot)
                if (
                    restored.compiled_composition is None
                    or not _world_matches(self.protocol, model, restored.compiled_composition.spec)
                    or snapshot["configuration"]["observation_noise_mode"] != self.noise_mode
                ):
                    raise ValueError("completed native snapshot differs from its protocol model")
        if self.complete:
            self.protocol.reports.evaluate(self.arrays)
        return self

    def _restore_runner(self, payload):
        if payload is None:
            if self.stack:
                raise ValueError("checkpoint task stack has no native task runner")
            return
        if (
            set(payload) != {"root", "sessions", "epochs", "readings", "history", "iterations"}
            or self.root_index >= len(self.roots)
            or payload["root"] != self.roots[self.root_index]
        ):
            raise ValueError("checkpoint native runner does not match its root")
        runner = _TaskRunner(
            self.protocol, payload["root"], self.seed, self.noise_mode, self.identity_namespace
        )
        involved = self.protocol.tasks.model_ids[payload["root"]]
        if (
            set(payload["sessions"]) != set(payload["epochs"])
            or set(payload["sessions"]) - involved
            or set(payload["readings"]) - involved
        ):
            raise ValueError("checkpoint sessions must belong to the active task's models")
        for model, snapshot in payload["sessions"].items():
            epoch = payload["epochs"][model]
            if type(epoch) is not int or epoch < 1:
                raise ValueError("checkpoint model epoch must be a positive integer")
            runtime = ChemWorldRuntime.from_snapshot(snapshot)
            if runtime.compiled_composition is None:
                raise ValueError("checkpoint needs the compiled native model world")
            world = runtime.compiled_composition.spec
            binding = f"{self.seed}:{payload['root']}:{model}:{epoch - 1}".encode()
            expected_seed = int.from_bytes(hashlib.sha256(binding).digest()[:4], "big")
            expected_identity = (
                "episode-"
                + uuid5(UUID(self.identity_namespace), f"{payload['root']}:{model}:{epoch - 1}").hex
            )
            if (
                runtime.seed != expected_seed
                or snapshot["continuation"]["_campaign_id"] != expected_identity
                or not _world_matches(self.protocol, model, world)
                or snapshot["configuration"]["observation_noise_mode"] != self.noise_mode
            ):
                raise ValueError("checkpoint native model/seed/noise policy differs from protocol")
            runner.runtimes[model], runner.worlds[model] = runtime, world
        runner.epochs, runner._readings = payload["epochs"], payload["readings"]
        runner.history = [HistoryRecord(**record) for record in payload["history"]]
        runner.iterations = payload["iterations"]
        self.runner = runner

    def _validate_stack(self) -> None:
        if len(self.stack) > 16 or bool(self.stack) != (self.runner is not None):
            raise ValueError("checkpoint stack depth or native runner presence is invalid")
        courses = sum(
            self.protocol.tasks.invocations[root] for root in self.roots[: self.root_index]
        )
        for depth, frame in enumerate(self.stack):
            if (
                set(frame) != {"task", "phase", "index", "sub_index", "ranges", "parts"}
                or frame["task"] not in self.protocol.tasks.nodes
            ):
                raise ValueError("checkpoint has unknown task frame fields or identity")
            if frame["task"] not in self.protocol.tasks.ranges:
                if self.status == "failed" and frame["phase"] == "start":
                    continue  # a failed simple course has not returned any arrays
                raise ValueError("checkpoint task frame is unknown or not at a course boundary")
            task = frame["task"]
            children = self.protocol.tasks.sub_tasks[task]
            if (
                type(frame["index"]) is not int
                or type(frame["sub_index"]) is not int
                or not 0 <= frame["index"] < self.protocol.tasks.ranges[task].count
                or not 0 <= frame["sub_index"] <= len(children)
                or frame["phase"] not in {"subtask", "collect"}
                or len(frame["parts"]) != frame["index"] * len(children) + frame["sub_index"]
            ):
                if self.status == "failed":
                    continue
                raise ValueError("checkpoint task positions/collected outputs are inconsistent")
            range_ids = {
                node.attributes["id"]
                for node in records(self.protocol.tasks.nodes[task], "listOfRanges")
            }
            if set(frame["ranges"]) != range_ids or any(
                type(value) not in (int, float) or not isfinite(value)
                for value in frame["ranges"].values()
            ):
                raise ValueError("checkpoint must retain all finite declared range values")
            if (
                self.status != "failed"
                and depth == len(self.stack) - 1
                and frame["phase"] != "subtask"
            ):
                raise ValueError("checkpoint top frame is waiting for a missing child")
            for index, part in enumerate(frame["parts"]):
                child_id = children[index % len(children)].attributes["task"]
                layouts = self.protocol.layouts[child_id]
                if set(part) != set(layouts):
                    raise ValueError("checkpoint collected output has different model identities")
                for model, channels in part.items():
                    spec = self.protocol.worlds[model].physics.models_of("sbml")["formal_workspace"]
                    units = {
                        observable.channel_id: observable.unit for observable in spec.observables
                    }
                    units["$time"] = "s"
                    if set(channels) != set(units) or any(
                        (array.shape, array.axes) != layouts[model] or array.unit != units[key]
                        for key, array in channels.items()
                    ):
                        raise ValueError("checkpoint collected channel layout or units differ")
                    prefix = (
                        "repeat:" + task,
                        index // len(children),
                        "subtask",
                        index % len(children),
                    )
                    if any(
                        sample[:4] != prefix
                        for array in channels.values()
                        for sample in array.sample_keys
                    ):
                        raise ValueError(
                            "checkpoint output coordinates do not match task positions"
                        )
            if depth == 0:
                if task != self.roots[self.root_index]:
                    raise ValueError("checkpoint stack starts at the wrong root")
            else:
                parent = self.stack[depth - 1]
                parent_children = self.protocol.tasks.sub_tasks[parent["task"]]
                if (
                    parent["phase"] != "collect"
                    or parent["sub_index"] >= len(parent_children)
                    or parent_children[parent["sub_index"]].attributes["task"] != task
                ):
                    raise ValueError("checkpoint stack path differs from the compiled task graph")
            courses += frame["index"] * sum(
                self.protocol.tasks.invocations[c.attributes["task"]] for c in children
            )
            courses += sum(
                self.protocol.tasks.invocations[c.attributes["task"]]
                for c in children[: frame["sub_index"]]
            )
        if self.status != "failed" and courses != self.courses_completed:
            raise ValueError("checkpoint completed course count differs from task positions")
