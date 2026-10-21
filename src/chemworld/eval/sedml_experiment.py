"""Execute declared experiment steps through the public native session API.

The adapter reads public laboratory time and declared Observation channels.
It does not read coordinates, integrate equations, or retain an evolving model
clock. Historical sample coordinates and native HistoryRecords are outputs.
"""

import hashlib
from dataclasses import dataclass, replace
from types import MappingProxyType

from chemworld.agents.base import HistoryRecord
from chemworld.data.sedml_declarations import boolean, records
from chemworld.data.sedml_models import calculation
from chemworld.data.sedml_ranges import _range_reference
from chemworld.data.sedml_reports import SEDMLArray
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sedml_protocol import SEDMLProtocol, parameter_target


class SEDMLExecutionError(RuntimeError):
    def __init__(self, message, *, history, runtime):
        super().__init__(message)
        self.history = tuple(history)
        self.snapshot = runtime.snapshot().to_dict()


@dataclass(frozen=True)
class SEDMLCourseResult:
    runtime: ChemWorldRuntime
    native_channels: object
    history: tuple
    initial_model_time: float

    def source_array(self, quantity):
        key = "$time" if quantity.observable is None else quantity.observable.channel_id
        array = self.native_channels[key]
        values = tuple(self.initial_model_time+value*quantity.native_to_source
                       if quantity.observable is None else value*quantity.native_to_source
                       for value in array.values)
        return SEDMLArray(values, array.shape, array.axes, array.sample_keys, quantity.source_unit)


def run_uniform_course(protocol, task_id, *, runtime=None, seed=0,
                       observation_noise_mode="sequential"):
    """Run one simple task, optionally continuing the same native model session.

The source grid specifies durations relative to this invocation's start. The
actual source/model clock continues across invocations, including warmup.
"""
    if not isinstance(protocol, SEDMLProtocol) or task_id not in protocol.tasks.nodes:
        raise ValueError("course execution requires a declared task in a compiled native protocol")
    task = protocol.tasks.nodes[task_id]
    if task.local_name != "task":
        raise ValueError("uniform course execution requires a simple task")
    model_id = task.attributes["modelReference"]
    world = protocol.worlds[model_id]
    simulation = protocol.tasks.simulations[task.attributes["simulationReference"]]
    return _run_course(world, simulation, runtime=runtime, seed=seed,
                       observation_noise_mode=observation_noise_mode)


def _run_course(world, simulation, *, runtime=None, seed=0,
                observation_noise_mode="sequential"):
    if runtime is None:
        runtime = ChemWorldRuntime(composition=world, seed=seed,
                                   observation_noise_mode=observation_noise_mode)
        runtime.reset(seed=seed)
    elif (not isinstance(runtime, ChemWorldRuntime) or runtime.compiled_composition is None
            or runtime.compiled_composition.spec.to_dict() != world.to_dict()):
        raise ValueError("continued native session does not match the compiled model declaration")
    spec = world.physics.models_of("sbml")["formal_workspace"]
    scale = spec.model.model_unit("timeUnits").time_s(1.)
    started_at = runtime.lab_status()["time_s"]
    channels = {o.channel_id: [] for o in spec.observables}
    channels["$time"] = []
    expected_units = {o.channel_id: o.unit for o in spec.observables}
    history, keys = [], []

    def execute(action):
        result = runtime.execute(action)
        reading = runtime.observe()
        history.append(HistoryRecord(len(history)+1, dict(action),
            {channel.channel_id: channel.value for channel in reading.channels},
            result.reward, dict(result.info), public_view=reading.native_record()))
        if result.info["operation_outcome"] != "success":
            raise SEDMLExecutionError("native experiment operation failed: "+str(action),
                                      history=history, runtime=runtime)

    try:
        for target in simulation.source_times:
            requested_time = started_at+(target-simulation.initial_time)*scale
            duration = requested_time-runtime.lab_status()["time_s"]
            if duration < 0:
                raise ValueError("source output grid would move the native clock backwards")
            if duration:
                execute({"operation": "wait", "duration_s": duration})
            if expected_units:
                execute({"operation": "measure", "instrument": "formal_quantity_meter"})
                reading = runtime.observe()
                if {c.channel_id: c.unit for c in reading.channels} != expected_units:
                    raise ValueError("native reading differs from declared channel units")
                for channel in reading.channels:
                    channels[channel.channel_id].append(channel.value)
            time_s = runtime.lab_status()["time_s"]
            channels["$time"].append(time_s)
            keys.append((spec.initial_model_time*scale+time_s,))
    except SEDMLExecutionError:
        raise
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        raise SEDMLExecutionError(str(exc), history=history, runtime=runtime) from exc
    arrays = {key: SEDMLArray(tuple(values), (len(keys),), ("time",), tuple(keys),
                "s" if key == "$time" else expected_units[key]) for key, values in channels.items()}
    return SEDMLCourseResult(runtime, MappingProxyType(arrays), tuple(history),
                              spec.initial_model_time)


@dataclass(frozen=True)
class SEDMLExperimentResult:
    reports: dict
    history: tuple
    iterations: tuple
    snapshots: dict


class _TaskRunner:
    """Finite protocol orchestration; every physical effect is a native action."""

    def __init__(self, protocol, root_id, seed, noise_mode):
        self.protocol, self.root_id = protocol, root_id
        self.seed, self.noise_mode = seed, noise_mode
        self.runtimes, self.worlds, self.epochs = {}, {}, {}
        self.history, self.iterations = [], []
        self._readings = {}

    def _history(self, history, model_id, task_id):
        for record in history:
            self.history.append(replace(record, step=len(self.history)+1,
                decision_context={"root_task": self.root_id, "model_reference": model_id,
                                  "task_reference": task_id}))

    def reset(self, model_id, world=None):
        world = self.protocol.worlds[model_id] if world is None else world
        epoch = self.epochs.get(model_id, 0)
        binding = f"{self.seed}:{self.root_id}:{model_id}:{epoch}".encode()
        seed = int.from_bytes(hashlib.sha256(binding).digest()[:4], "big")
        runtime = ChemWorldRuntime(composition=world, seed=seed,
                                   observation_noise_mode=self.noise_mode)
        runtime.reset(seed=seed)
        self.runtimes[model_id], self.worlds[model_id] = runtime, world
        self.epochs[model_id] = epoch+1
        self._readings.pop(model_id, None)

    def session(self, model_id):
        if model_id not in self.runtimes:
            self.reset(model_id)
        return self.runtimes[model_id]

    def execute(self, model_id, task_id, action):
        runtime = self.session(model_id)
        result = runtime.execute(action)
        reading = runtime.observe()
        record = HistoryRecord(0, dict(action),
            {channel.channel_id: channel.value for channel in reading.channels},
            result.reward, dict(result.info), public_view=reading.native_record())
        self._history((record,), model_id, task_id)
        self._readings.pop(model_id, None)
        if result.info["operation_outcome"] != "success":
            raise SEDMLExecutionError("native scan operation failed: "+str(action),
                                      history=self.history, runtime=runtime)

    def model_value(self, variable, *, shadow=None):
        model_id, quantity = self.protocol.quantities[variable.attributes["id"]]
        observable = quantity.observable
        if (shadow is not None and observable is not None and observable.kind == "parameter"
                and (model_id, observable.source_id) in shadow):
            return shadow[model_id, observable.source_id]
        runtime = self.session(model_id)
        if observable is not None and model_id not in self._readings:
            self.execute(model_id, self.root_id,
                         {"operation": "measure", "instrument": "formal_quantity_meter"})
            self._readings[model_id] = {c.channel_id: c.value for c in runtime.observe().channels}
        spec = self.worlds[model_id].physics.models_of("sbml")["formal_workspace"]
        return quantity.value(self._readings.get(model_id, {}),
            time_s=runtime.lab_status()["time_s"], initial_model_time=spec.initial_model_time)

    def changes(self, nodes, ranges, task_id, *, initialize=False):
        self._readings.clear()
        shadow, targets = {}, {}

        def variable_value(variable):
            reference = _range_reference(variable)
            return (ranges[reference] if reference is not None else
                    self.model_value(variable, shadow=shadow))

        for node in nodes:
            model_id = node.attributes["modelReference"]
            target = parameter_target(self.protocol.models[model_id], node)
            reference = node.attributes.get("range")
            value = calculation(node, variable_value=variable_value,
                extra={} if reference is None else {reference: ranges[reference]})
            shadow[model_id, target] = value
            targets.setdefault(model_id, []).append(node.attributes["target"].endswith("/@value"))
        for model_id in sorted(targets):
            values = {name: value for (model, name), value in shadow.items() if model == model_id}
            if initialize and all(targets[model_id]):
                self.reset(model_id, self.protocol.world_with_parameters(model_id, values))
            else:
                self.execute(model_id, task_id,
                             {"operation": "set_model_parameters", "values": values})

    def task(self, identity):
        node = self.protocol.tasks.nodes[identity]
        if node.local_name == "task":
            model_id = node.attributes["modelReference"]
            runtime = self.session(model_id)
            simulation = self.protocol.tasks.simulations[node.attributes["simulationReference"]]
            try:
                result = _run_course(self.worlds[model_id], simulation, runtime=runtime)
            except SEDMLExecutionError as exc:
                self._history(exc.history, model_id, identity)
                exc.history = tuple(self.history)
                raise
            self._history(result.history, model_id, identity)
            return {model_id: result.native_channels}
        parts = []
        reset = boolean(node.attributes["resetModel"])
        for index in range(self.protocol.tasks.ranges[identity].count):
            if reset:
                for model_id in sorted(self.protocol.tasks.model_ids[identity]):
                    self.reset(model_id)
            self._readings.clear()
            ranges = self.protocol.tasks.ranges[identity].values_at(index,
                                                                   model_value=self.model_value)
            self.iterations.append({"task_reference": identity, "index": index,
                                     "range_values": dict(ranges)})
            self.changes(records(node, "listOfChanges"), ranges, identity, initialize=reset)
            for sub_index, child in enumerate(self.protocol.tasks.sub_tasks[identity]):
                self.changes(records(child, "listOfChanges"), ranges, identity)
                result = self.task(child.attributes["task"])
                prefix = ("repeat:"+identity, index, "subtask", sub_index,
                          *(value for pair in sorted(ranges.items()) for value in pair))
                parts.append({model: {key: SEDMLArray(array.values, array.shape, array.axes,
                    tuple((*prefix, *sample) for sample in array.sample_keys), array.unit)
                    for key, array in arrays.items()} for model, arrays in result.items()})
        result = {}
        for model_id, (shape, axes) in self.protocol.layouts[identity].items():
            model_parts = [part[model_id] for part in parts if model_id in part]
            result[model_id] = {key: SEDMLArray(
                tuple(v for part in model_parts for v in part[key].values), shape, axes,
                tuple(sample for part in model_parts for sample in part[key].sample_keys),
                model_parts[0][key].unit) for key in model_parts[0]}
        return result


def run_sedml_experiment(protocol, *, seed=0, observation_noise_mode="sequential"):
    """Execute all requested task outputs, preserving independent task roots."""
    if (not isinstance(protocol, SEDMLProtocol) or type(seed) is not int or seed < 0
            or observation_noise_mode not in ("sequential", "keyed")):
        raise ValueError("experiment requires a compiled protocol, seed and native noise mode")
    selected = {identity for identity, _, _ in protocol.reports.bindings.values()}
    if not selected:
        selected = set(protocol.tasks.roots)
    arrays, history, iterations, snapshots = {}, [], [], {}
    for identity in protocol.tasks.nodes:
        if identity not in selected:
            continue
        runner = _TaskRunner(protocol, identity, seed, observation_noise_mode)
        try:
            channels = runner.task(identity)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            saved = {**snapshots, identity: {model: session.snapshot().to_dict()
                     for model, session in runner.runtimes.items()}}
            failed_history = (*history, *(replace(record, step=len(history)+i+1)
                for i, record in enumerate(runner.history)))
            error = SEDMLExecutionError(str(exc), history=failed_history,
                                        runtime=next(reversed(runner.runtimes.values())))
            if isinstance(exc, SEDMLExecutionError):
                error.snapshot = exc.snapshot
            error.snapshots = saved
            error.iterations = (*iterations, *runner.iterations)
            raise error from exc
        for key, (task_id, model_id, quantity) in protocol.reports.bindings.items():
            if task_id != identity:
                continue
            spec = protocol.worlds[model_id].physics.models_of("sbml")["formal_workspace"]
            course = SEDMLCourseResult(runner.runtimes[model_id], channels[model_id], (),
                                        spec.initial_model_time)
            arrays[key] = course.source_array(quantity)
        for record in runner.history:
            history.append(replace(record, step=len(history)+1))
        iterations.extend(runner.iterations)
        snapshots[identity] = {model: session.snapshot().to_dict()
                               for model, session in runner.runtimes.items()}
    try:
        reports = protocol.reports.evaluate(arrays)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        error = SEDMLExecutionError(str(exc), history=history,
                                    runtime=next(reversed(runner.runtimes.values())))
        error.snapshots, error.iterations = snapshots, tuple(iterations)
        error.variable_arrays = {str(key): array.to_dict() for key, array in arrays.items()}
        raise error from exc
    return SEDMLExperimentResult(reports, tuple(history), tuple(iterations), snapshots)
